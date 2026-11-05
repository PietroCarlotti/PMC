# Test arithmetic-mean consensus
test_that("mean consensus averages components", {
  # Build two draws from two bivariate components
  samples <- array(0, dim = c(2, 2, 2))
  samples[, 1, ] <- matrix(c(1, 2, 3, 4), nrow = 2, byrow = TRUE)
  samples[, 2, ] <- matrix(c(5, 6, 7, 8), nrow = 2, byrow = TRUE)

  # Average each component pair
  expected <- matrix(c(3, 4, 5, 6), nrow = 2, byrow = TRUE)

  expect_equal(consensus_mean(samples), expected)
})

# Test matrix-weighted consensus
test_that("weighted consensus applies matrix weights", {
  # Build two draws from two bivariate components
  samples <- array(0, dim = c(2, 2, 2))
  samples[, 1, ] <- matrix(c(1, 2, 3, 4), nrow = 2, byrow = TRUE)
  samples[, 2, ] <- matrix(c(5, 6, 7, 8), nrow = 2, byrow = TRUE)

  # Give the second component three times the weight of the first
  weights <- array(0, dim = c(2, 2, 2))
  weights[1, , ] <- 0.25 * diag(2)
  weights[2, , ] <- 0.75 * diag(2)

  # Compute the expected weighted averages
  expected <- matrix(c(4, 5, 6, 7), nrow = 2, byrow = TRUE)

  expect_equal(consensus_weighted_average(samples, weights), expected)
})

# Test sample-shape validation
test_that("consensus rejects invalid sample shapes", {
  # Pass a matrix instead of an S by K by p array
  expect_error(
    consensus_mean(matrix(1:4, nrow = 2)),
    "dimensions c\\(S, K, p\\)"
  )
})
