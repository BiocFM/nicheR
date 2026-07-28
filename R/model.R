#' Fill masked padding positions and derive the attention mask
#'
#' Remaps padding tokens (0 -> `padding_token`) in a batch's token tensor
#' and derives a boolean `attention_mask` marking padded positions, as
#' expected by [nicheformer()]'s `get_embeddings()`.
#'
#' @param batch a list with element `X`, an integer token tensor.
#' @return `batch` with `masked_indices` (remapped tokens) and
#'   `attention_mask` (logical tensor, `TRUE` at padding positions) added.
#' @noRd
complete_masking_p0 <- function(batch) {
  padding_token <- 1L
  x <- batch[["X"]]
  # [IDX] reference converts padding 0 -> 1
  x <- torch_where(x == 0L, torch_tensor(padding_token, dtype = x$dtype, device = x$device), x)
  batch[["masked_indices"]] <- x
  batch[["attention_mask"]] <- (x == padding_token)$to(dtype = torch_bool())
  batch
}

#' Build a NicheFormer model and load pretrained weights
#'
#' Constructs an R `torch` transformer-encoder replica of the python
#' NicheFormer architecture (theislab/nicheformer), loads the pretrained
#' state dict bundled with the package, and returns it ready for
#' inference via [run_nicheformer()].
#'
#' @param weights model weights trained by a nicheformer model. 
#' @param dim_model embedding/model dimension (`d_model`) of the
#'   transformer encoder.
#' @param nheads number of attention heads per encoder layer.
#' @param dim_feedforward size of the hidden state of the feedforward
#'   network within each encoder layer.
#' @param nlayers number of stacked transformer encoder layers.
#' @param dropout node dropout probability in the feedforward network
#'   (relevant for training; the returned model is set to `eval()` mode).
#' @param batch_first logical; if `TRUE` (default) tensors are shaped
#'   `(batch, sequence, feature)`.
#' @param n_tokens vocabulary size (number of distinct gene/context
#'   tokens).
#' @param context_length maximum sequence length (number of token
#'   positions), used to size the positional embedding.
#' @param learnable_pe logical; if `TRUE` use a learnable positional
#'   embedding, otherwise derive positions from the token embedding.
#' @param dev device on which to place the model, e.g. `"cpu"` or a CUDA
#'   device string such as `"cuda"`.
#' @return a `torch::nn_module` instance in evaluation mode, with the
#'   pretrained weights from the packaged `nicheformer_embedpath.pt`
#'   state dict loaded and moved to `dev`.
#' @import torch
#' @import safetensors
#' @export
nicheformer <- function(
  weights,
  dim_model = 512L,
  nheads = 16L, 
  dim_feedforward = 1024L,
  nlayers = 12L, 
  dropout = 0.0, 
  batch_first = TRUE,
  n_tokens = 20340L, 
  context_length = 1500L,
  learnable_pe = TRUE,
  dev = "cpu"
){
  nicheformer <- nn_module(
    "NicheFormer",
    initialize = function() {
      # store what forward/get_embeddings need (no self$hparams in R torch)
      self$learnable_pe   <- learnable_pe
      self$context_length <- context_length
      self$n_tokens       <- n_tokens

      # --- transformer ---
      encoder_layer <- nn_transformer_encoder_layer(
        d_model = dim_model, nhead = nheads,
        dim_feedforward = dim_feedforward, dropout = dropout,
        activation = "relu",              # reference default (NOT gelu)
        layer_norm_eps = 1e-12,
        batch_first = batch_first,        # once, not twice
        norm_first = FALSE
      )
      self$encoder <- nn_transformer_encoder(
        encoder_layer, num_layers = nlayers
      )

      # --- embeddings ---
      self$embeddings <- nn_embedding(
        num_embeddings = n_tokens + 5L, embedding_dim = dim_model,
        padding_idx = 2L                  # [IDX] py padding_idx=1 -> R row 2
      )
      self$positional_embedding <- nn_embedding(context_length, dim_model)
      self$dropout <- nn_dropout(p = dropout)
      # [IDX] py arange(0, L) selects rows 0..L-1; in 1-based R those rows are 1..L
      self$pos <- nn_buffer(torch_arange(start = 1, end = context_length,
                                        dtype = torch_long()))
    },

    get_embeddings = function(batch, with_context = FALSE) {
      batch <- complete_masking_p0(batch)
      masked_indices <- batch[["masked_indices"]]
      attention_mask <- batch[["attention_mask"]]

      # [IDX] py token id k -> R embedding row k+1
      token_embedding <- self$embeddings(masked_indices + 1L)

      if (self$learnable_pe) {
        pos_embedding <- self$positional_embedding(self$pos)
        embeddings <- self$dropout(token_embedding + pos_embedding$unsqueeze(1))
      } else {
        embeddings <- self$positional_embedding(token_embedding)
      }

      # manual per-layer loop (NOT self$encoder(...)) to avoid the nested-tensor
      # fast path that drops padding rows. Run ALL layers.
      n_layers <- length(self$encoder$layers)
      for (i in seq_len(n_layers)) {                     # 1..n_layers, no drop
        embeddings <- self$encoder$layers[[i]](
          embeddings, src_key_padding_mask = attention_mask
        )
      }

      if (!with_context) {
        # [IDX] py [:, 3:, :] = drop first 3, keep rest. SLICE, not select.
        embeddings <- embeddings[, 4:embeddings$size(2), ]
      }
      embeddings$mean(dim = 2)            # mean over sequence (R dim 2 = py dim 1)
    }
  )

  model <- nicheformer()

  W <- load_state_dict(weights)
  sd <- model$state_dict()

  with_no_grad({
    for (nm in names(W)) {
      if (nm %in% names(sd)) {
        sd[[nm]]$copy_(W[[nm]])
      } 
    }
  })

  model$load_state_dict(sd)

  dev <- dev
  model$to(device = dev)
  model$eval()

  return(model)
}

#' Compute NicheFormer embeddings for a token matrix
#'
#' Runs a fitted NicheFormer `model` (see [nicheformer()]) over `X` in
#' mini-batches with no gradient tracking, printing progress, and returns
#' the pooled embedding for every row (cell).
#'
#' @param model an `nn_module` produced by [nicheformer()].
#' @param X integer matrix of tokens, one row per cell, as produced by
#'   [tokenization()].
#' @param batch_size number of cells processed per forward pass.
#' @param device device used for computation, e.g. `"cpu"` or a CUDA
#'   device string; should match the device `model` was placed on.
#' @return a numeric matrix with one row per cell (same order as `X`)
#'   and `dim_model` columns, holding the mean-pooled embedding for each
#'   cell.
#' @export
run_nicheformer <- function(model, X, batch_size = 8L, device = "cpu") {
  n <- nrow(X)
  out <- vector("list", ceiling(n / batch_size))
  k <- 1L
  
  with_no_grad({
    for (start in seq(1L, n, by = batch_size)) {
      end <- min(start + batch_size - 1L, n)
      xb <- torch_tensor(X[start:end, , drop = FALSE],
                         dtype = torch_long(), device = device)
      e <- model$get_embeddings(list(X = xb), with_context = FALSE)
      out[[k]] <- as.matrix(e$cpu())     # pull to CPU immediately, free GPU mem
      k <- k + 1L
      rm(xb, e); gc()
      cat(sprintf("\r%d / %d cells", end, n))
      if(torch::cuda_is_available()) cuda_empty_cache()
    }
  })
  cat("\n")
  do.call(rbind, out)
}


