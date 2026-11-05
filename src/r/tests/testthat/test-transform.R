# Test identity and moment transforms
test_that("moment transforms preserve expected shapes and values", {
  # Build two bivariate consensus values
  consensus <- matrix(c(1, 2, -1, 3), nrow = 2, byrow = TRUE)

  expect_equal(transform_identity(consensus), consensus)
  expect_equal(
    transform_hadamard_product(consensus),
    matrix(c(1, 4, 1, 9), nrow = 2, byrow = TRUE)
  )
  expect_equal(transform_inner_product(consensus), matrix(c(5, 10)))
  expect_equal(
    transform_outer_product(consensus),
    matrix(c(1, 2, 2, 4, 1, -3, -3, 9), nrow = 2, byrow = TRUE)
  )
})

# Test indicator boundaries
test_that("indicator transforms use the expected boundaries", {
  # Build two bivariate consensus values
  consensus <- matrix(c(1, 2, -1, 3), nrow = 2, byrow = TRUE)

  expect_equal(
    transform_indicator_threshold(consensus, c(0, 2)),
    matrix(c(1, 0, 0, 1), nrow = 2, byrow = TRUE)
  )
  expect_equal(
    transform_indicator_ball(consensus, c(0, 0), sqrt(5)),
    matrix(c(1, 0))
  )
  expect_equal(
    transform_indicator_box(consensus, c(-1, 2), c(1, 3)),
    matrix(1, nrow = 2, ncol = 2)
  )
})

# Test indicator-dimension validation
test_that("indicator transforms reject mismatched dimensions", {
  # Pass one threshold for a bivariate consensus
  consensus <- matrix(c(1, 2), nrow = 1)

  expect_error(
    transform_indicator_threshold(consensus, 0),
    "one value per consensus coordinate"
  )
})
