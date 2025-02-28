function loss = obiectiv(e, y)
    N = length(e);
    loss = - (1 / N) * sum(e .* log(y) + (1 - e) .* log(1 - y));
end
