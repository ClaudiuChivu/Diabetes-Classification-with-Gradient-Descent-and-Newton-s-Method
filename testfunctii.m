% Setăm datele
N = 100;
n = 5;
m = 5;

A = randn(N, n);
X = randn(n, m);
x = randn(m, 1);
e = randn(N, 1);

% Calculăm ieșirea rețelei
y = f_activare(A * X )*x;

% Calculăm gradientul folosind funcția noastră
gradient = gradientulll(e, y, X, A);
hess=hessiana(e,y,X,A);
diag_y = diag(y .* (1 - y))

% Afișăm gradientul
disp('Gradient:');
disp(gradient);
disp('hes:');
disp(hess);
disp(norm(gradient));
loss=obiectiv(e,y);



