#' Prepend context tokens to a matrix of gene-based tokens
#'
#' Adds three leading columns (species, assay, modality context tokens)
#' to a per-cell matrix of ranked gene tokens, as expected by
#' [nicheformer()]'s input format.
#'
#' @param tokens integer matrix of gene-based tokens, one row per cell.
#' @param specie context token for species.
#' @param assay_tok context token for assay/technology.
#' @param modality context token for modality.
#' @return an integer matrix with the same rows as `tokens`, and 3 extra
#'   leading columns holding `specie`, `assay_tok`, and `modality`.
#' @noRd
prepend_context <- function(tokens, specie, assay_tok, modality) {
  n <- nrow(tokens)
  cbind(rep(specie, n), rep(assay_tok, n), rep(modality, n), tokens)
}

#' Rank and tokenize a single cell's gene expression
#'
#' Applies technology-mean correction to one cell's counts, ranks genes by
#' corrected expression, maps the top genes to reference vocabulary
#' positions, and zero-pads/truncates the result to a fixed length.
#'
#' @param counts numeric vector of raw gene counts for one cell, aligned
#'   to `tech_mean` and `ref_pos0` (same gene panel, same order).
#' @param tech_mean numeric vector of mean expression per gene for the
#'   relevant technology, aligned to `counts`.
#' @param ref_pos0 integer vector of 0-based reference-vocabulary
#'   positions for each gene, aligned to `counts`; `NA` for genes absent
#'   from the reference.
#' @param L total number of gene-token slots to fill.
#' @param AUX number of auxiliary/context tokens reserved before the
#'   gene vocabulary (added as an offset to `ref_pos0`).
#' @return an integer vector of length `L` holding the ranked, offset
#'   gene tokens, zero-padded if fewer than `L` genes are available.
#' @noRd
tokenize_cell <- function(counts, tech_mean, ref_pos0, L, AUX = 30L) {
  # all three args are PANEL-aligned, same length

  keep <- !is.na(ref_pos0)          # drop genes absent from the reference
  counts    <- counts[keep]
  tech_mean <- tech_mean[keep]
  ref_pos0  <- ref_pos0[keep]

  s <- sum(counts); if (s == 0) s <- 1
  x <- counts * (10000/s)
  tech_mean[tech_mean == 0] <- 1        # zeros only, per reference
  x <- x / tech_mean                     # NaNs propagate
  nz <- which(x != 0 | is.na(x))         # include NaN, like np.nonzero
  ord <- order(-x[nz], na.last = TRUE, method = "radix")   # NaN last
  ranked <- nz[ord]
  if (length(ranked) > L) ranked <- ranked[1:L]

  tokens <- ref_pos0[ranked] + AUX  # ref_pos0 already 0-based -> no -1 here

  scores <- integer(L)
  if (length(tokens) > 0) scores[seq_along(tokens)] <- tokens
  return(scores)
}

#' Tokenize a spatial single-cell experiment for NicheFormer
#'
#' Ranks genes per cell by technology-corrected expression, maps them to
#' reference vocabulary positions, zero-pads/truncates to a fixed length,
#' and prepends the species/assay/modality context tokens required by
#' [nicheformer()]'s `get_embeddings()`.
#'
#' @param counts a `SingleCellExperiment` with gene counts in its first
#'   assay, whose `rownames()` are (a subset of) the `refsce` reference
#'   gene panel bundled with the package.
#' @param tech_means an object of technology specific means. 
#' @param specie integer context token identifying species.
#' @param assay_tok integer context token identifying the assay/technology.
#' @param modality integer context token identifying the modality.
#' @param CONTEXT_LENGTH total sequence length expected by the model,
#'   including the 3 leading context tokens.
#' @importFrom utils data
#' @import SingleCellExperiment
#' @return an integer matrix with one row per cell and `CONTEXT_LENGTH`
#'   columns: 3 leading context tokens (species, assay, modality) followed
#'   by ranked, technology-corrected gene tokens, zero-padded as needed.
#' @export
tokenization <- function(counts, tech_means, specie, assay_tok, modality,CONTEXT_LENGTH = 1500L){
  utils::data("refsce", package="nicheR")
  names(tech_means) <- rownames(refsce)
  tech_means <-  tech_means[match(rownames(counts),names(tech_means))]
  panel_to_ref0 <- match(rownames(counts), rownames(refsce)) - 1L

  n_gene_slots <- CONTEXT_LENGTH - 3L   # 3 reserved for context tokens

  tokens <- t(vapply(seq_len(ncol(counts)),
    function(i) tokenize_cell(assay(counts)[, i], tech_means, panel_to_ref0,
                              L = n_gene_slots),
    integer(n_gene_slots)))

  return(prepend_context(tokens,specie, assay_tok, modality))
}
