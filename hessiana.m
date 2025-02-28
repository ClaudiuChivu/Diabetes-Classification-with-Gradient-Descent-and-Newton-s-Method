function hess = hessiana(e, y, X, A)
    N = length(e);
    diag_y = diag(y .* (1 - y));
    hess = (A' * diag_y * A) / N;
end
