function gradient = gradientulll(e, y, X, A)
    N = length(e);
    gradient = zeros(size(X, 2), 1);
    for i = 1:N
        gradient = gradient + (e(i) - y(i)) * A(i);
    end
    gradient = -gradient' / N;
end
