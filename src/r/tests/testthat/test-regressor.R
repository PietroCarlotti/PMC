# Test every OLS fitting branch
test_that("OLS recovers the intercept across draw regimes", {
  # Use ample draws for centered normal equations
  discrepancy <- matrix(-2:3, ncol = 1)
  target <- matrix(3 + 2 * discrepancy[, 1], ncol = 1)
  expect_equal(regressor_ols(discrepancy, target), 3)

  # Use just enough draws for the ordinary least-squares branch
  discrepancy <- matrix(c(-1, 1), ncol = 1)
  target <- matrix(c(1, 5), ncol = 1)
  expect_equal(regressor_ols(discrepancy, target), 3)

  # Use fewer draws than parameters for minimum-norm slopes
  discrepancy <- matrix(c(1, 0, -1, 0), nrow = 2, byrow = TRUE)
  target <- matrix(c(6, 2), ncol = 1)
  expect_equal(regressor_ols(discrepancy, target), 4)
})

# Test regression-shape validation
test_that("OLS rejects mismatched draw counts", {
  # Pass different numbers of discrepancy and target draws
  discrepancy <- matrix(c(-1, 1), ncol = 1)
  target <- matrix(c(1, 3, 5), ncol = 1)

  expect_error(regressor_ols(discrepancy, target), "matching S")
})
