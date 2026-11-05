# Test anchored-difference ordering
test_that("anchored differences use component-first ordering", {
  # Build two draws from three bivariate components
  samples <- array(0, dim = c(2, 3, 2))
  samples[, 1, ] <- matrix(c(1, 10, 2, 20), nrow = 2, byrow = TRUE)
  samples[, 2, ] <- matrix(c(4, 16, 8, 28), nrow = 2, byrow = TRUE)
  samples[, 3, ] <- matrix(c(0, 7, 5, 25), nrow = 2, byrow = TRUE)

  # Flatten each component difference before moving to the next component
  expected <- matrix(
    c(3, 6, -1, -3, 6, 8, 3, 5),
    nrow = 2,
    byrow = TRUE
  )

  expect_equal(discrepancy_anchored_differences(samples), expected)
})

# Test finite and infinite norm orders
test_that("norm discrepancy reduces each difference vector", {
  # Reuse known anchored differences
  samples <- array(0, dim = c(2, 3, 2))
  samples[, 1, ] <- matrix(c(1, 10, 2, 20), nrow = 2, byrow = TRUE)
  samples[, 2, ] <- matrix(c(4, 16, 8, 28), nrow = 2, byrow = TRUE)
  samples[, 3, ] <- matrix(c(0, 7, 5, 25), nrow = 2, byrow = TRUE)

  expect_equal(discrepancy_norm(samples), matrix(sqrt(c(55, 134))))
  expect_equal(discrepancy_norm(samples, Inf), matrix(c(6, 8)))
})

# Test norm-order validation
test_that("norm discrepancy rejects orders below one", {
  # Pass an invalid norm order
  samples <- array(1, dim = c(1, 2, 1))

  expect_error(discrepancy_norm(samples, 0), "at least 1")
})
