# Test the default PMC pipeline
test_that("PMC recovers a known zero-discrepancy intercept", {
  # Construct consensus values that are linear in discrepancy
  discrepancy <- -3:3
  consensus <- 5 + 2 * discrepancy
  samples <- array(0, dim = c(length(discrepancy), 2, 1))
  samples[, 1, 1] <- consensus - discrepancy / 2
  samples[, 2, 1] <- consensus + discrepancy / 2

  expect_equal(pmc(samples), 5)
})

# Test custom PMC components
test_that("PMC composes custom transforms and regressors", {
  # Build univariate samples with known consensus values
  samples <- array(c(1, 3, 2, 4), dim = c(2, 2, 1))
  expected <- mean(consensus_mean(samples)^2)
  mean_regressor <- function(discrepancy, target) {
    colMeans(target)
  }

  estimate <- pmc(
    samples,
    h = transform_hadamard_product,
    regressor = mean_regressor
  )

  expect_equal(estimate, expected)
})
