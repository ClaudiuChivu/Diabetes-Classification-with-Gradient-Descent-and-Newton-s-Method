function y = iesirea_y(A, x, X)
    % A - matricea de intrări
    % x - vectorul de ponderi (parametrii modelului)
    % X - matricea de ponderi asociate neuronilor
    
    % Numărul de exemple
    N = size(A, 1);
    
    % Calculul ieșirii y pentru fiecare intrare
    y = zeros(N, 1);
    for i = 1:N
        % Calculul produsului scalar a_i^T * X
        a_i = A(i, :);
        z = a_i * X;
        
        % Aplicarea funcției tangente hiperbolice asupra produsului scalar
        g_z = f_activare(z);
        
        % Calculul ieșirii y folosind produsul scalar obținut și vectorul de ponderi
        y(i) = g_z * x;
    end
end
