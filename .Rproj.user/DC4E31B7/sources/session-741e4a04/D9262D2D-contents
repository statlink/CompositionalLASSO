cv.lasso.mziln <- function(y, x, lambda = NULL, nlambda = 100, nfolds = 10,
                             folds = NULL, seed = NULL) {

  runtime <- proc.time()
  if ( is.null( colnames(x) ) ) {
    p <- dim(x)[2]
    nama <- c("constant", paste("X", 1:p, sep = "") )
  } else  nama <- c("constant", colnames(x) )

  x <- cbind(1, x)
  n <- dim(y)[1]  ;  D <- dim(y)[2]  ;  d <- D - 1
  p <- dim(x)[2]  ;  M <- d * (p + 1)
  dd <- diag(d)

  sel1 <- Rfast::rowsums(y > 0) == D
  y1 <- y[sel1, ]
  ly <- log(y1[, -D] / y1[, D])
  x1 <- x[sel1, , drop = FALSE]
  y0 <- y[!sel1, ,drop = FALSE]
  x0 <- x[!sel1, , drop = FALSE]
  n1 <- dim(y1)[1]  ;  n0 <- dim(y0)[1]

  subj_list1 <- Filter( Negate(is.null), lapply( seq_len(n1), function(i) .build_1(ly[i, ], x1[i, ], dd) ) )
  subj_list0 <- Filter( Negate(is.null), lapply( seq_len(n0), function(i) .build_0(y0[i, ], x0[i, ], d, dd) ) )

  xtilde1 <- utilde1 <- list()
  for ( s in subj_list1 ) {
    xtilde1[[ length(xtilde1) + 1 ]] <- s$D
    utilde1[[ length(utilde1) + 1 ]] <- s$u
  }
  xtilde0 <- utilde0 <- list()
  for ( s in subj_list0 ) {
    Omega_i <- solve( s$A %*% t(s$A) )
    Omega_half <- chol(Omega_i)
    xtilde0[[ length(xtilde0) + 1 ]] <- Omega_half %*% s$D
    utilde0[[ length(utilde0) + 1 ]] <- Omega_half %*% s$u
  }

  xtilde <- do.call(c, list(xtilde1, xtilde0))
  utilde <- do.call(c, list(utilde1, utilde0))

  id <- numeric(n)
  sel1 <- which( Rfast::rowsums(y > 0) == D )
  id[sel1] <- 1
  if ( is.null(folds) )  folds <- Compositional::makefolds(id, nfolds = nfolds, stratified = TRUE, seed = seed)
  nfolds <- length(folds)

  X <- do.call(rbind, xtilde)
  U <- do.call(c, utilde)

  mod <- glmnet::glmnet(X, U, nlambda = nlambda, lambda = lambda, family = "gaussian")

  lambda <- mod$lambda
  nlambda <- length(mod$lambda)
  be <- list()

  for ( i in 1:nlambda ) {
    be[[ i ]] <- matrix(mod$beta[, i], ncol = d)
    colnames(be[[ i ]]) <- paste0("Y", 1:d)
    rownames(be[[ i ]]) <- nama
  }
  names(be) <- paste("lambda=", lambda, sep="")

  kld <- matrix(nrow = nfolds, ncol = nlambda )

  for ( k in 1:nfolds ) {

    train <- unlist( folds[ -k] )
    Xtrain <- do.call( rbind, xtilde[train] )
    Utrain <- do.call( c, utilde[train] )
    xtest <- x[ folds[[ k ]], ]
    ytest <- y[ folds[[ k ]], ]

    mod <- glmnet::glmnet(Xtrain, Utrain, nlambda = nlambda, lambda = lambda, family = "gaussian")
    len <- length(mod$lambda)
    ntest <- dim(ytest)[1]
    for ( j in 1:len ) {
      B <- matrix(mod$beta[, j], ncol = d)
      mu <- cbind( exp( xtest %*% B ), 1 )
      est <- mu / Rfast::rowsums(mu)
      kl <- ytest * log(ytest / est)
      kl[is.infinite(kl)] <- NA
      kld[k, j] <- sum(kl, na.rm = TRUE) / ntest
    }
  }

  kld <- cbind( lambda, Rfast::colmeans(kld) )
  colnames(kld) <- c("lambda", "KLD")
  lopt <- which.min(kld[, 2])
  opt <- kld[lopt, ]
  names(opt) <- c("lambda", "KLD")

  runtime <- proc.time - runtime

  list(runtime = runtime, kld = kld, opt = opt, be = be[[ lopt ]] )
}
