cv.lasso.compreg <- function(y, x, lambda = NULL, nlambda = 100, nfolds = 10,
                               folds = NULL, seed = NULL) {

  runtime <- proc.time()

  D <- dim(y)[2]   ;   d <- D - 1
  z <- Compositional::alr(y)
  n <- dim(y)[1]  ## sample size
  ina <- 1:n
  if ( is.null(folds) )  folds <- Compositional::makefolds(ina, nfolds = nfolds, stratified = FALSE, seed = seed)
  nfolds <- length(folds)

  if ( is.null( colnames(x) ) ) {
    p <- dim(x)[2]
    nama <- c("constant", paste("X", 1:p, sep = "") )
  } else  nama <- c("constant", colnames(x) )
  colnames(x) <- nama[-1]

  mod <- glmnet::glmnet(x, z, nlambda = nlambda, lambda = lambda, family = "mgaussian")

  lambda <- mod$lambda
  nlambda <- length(mod$lambda)
  be <- list()
  beta <- mod$beta
  a0 <- mod$a0

  for ( i in 1:nlambda ) {
    ini <- beta[[ 1 ]][, i]
    for ( j in 2:d )  ini <- cbind(ini, beta[[ j ]][, i])
    be[[ i ]] <- ini
    be[[ i ]] <- rbind(mod$a0[, i], ini)
    colnames(be[[ i ]]) <- paste0("Y", 2:D)
    rownames(be[[ i ]]) <- nama
  }
  names(be) <- paste("lambda=", lambda, sep="")

  kld <- matrix(nrow = nfolds, ncol = nlambda )

  for ( k in 1:nfolds ) {
    xtest <- x[ folds[[ k ]], ]
    xtrain <- x[ -folds[[ k ]], ]
    ytest <- y[ folds[[ k ]], ]
    ztrain <- z[ -folds[[ k ]], ]
    mod <- glmnet::glmnet(xtrain, ztrain, nlambda = nlambda, lambda = lambda, family = "mgaussian")
    est <- predict(mod, newx = xtest)
    ntest <- dim(ytest)[1]
    for ( j in 1:dim(est)[3] ) {
      yest <- Compositional::alrinv( est[, , j] )
      kl <- ytest * log(ytest / yest)
      kl[is.infinite(kl)] <- NA
      kld[k, j] <- sum(kl, na.rm = TRUE) / ntest
    }
  }

  kld <- cbind( lambda, Rfast::colmeans(kld) )
  colnames(kld) <- c("lambda", "KLD")
  lopt <- which.min(kld[, 2])
  opt <- kld[lopt, ]
  names(opt) <- c("lambda", "KLD")

  runtime <- proc.time() - runtime

  list(runtime = runtime, kld = kld, opt = opt, be = be[[ lopt ]] )

}
