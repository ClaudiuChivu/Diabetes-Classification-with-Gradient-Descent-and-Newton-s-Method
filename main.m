clc;
clear;
close all;

data = readmatrix('Diabetes.csv');

% Extragerea dimensiunilor și a datelor
[N, n] = size(data);

% am definit setul de antrenare
train_size = round(0.8 * N); % Aproximativ 80% din date pentru antrenare

% Crearea matricei A și vectorului e
A = data(:, 1:n-1); 
e = data(:, n); % Ultima coloană conține etichetele

% Crearea vectorului 1_N
one_N = ones(N, 1);

% Separarea datelor în seturi de antrenare și testare
train_data = data(1:train_size, :);
test_data = data(train_size+1:end, :);

% salvarea seturilor de date
mkdir train_folder
mkdir test_folder

csvwrite('train_folder/train_data.csv', train_data);
csvwrite('test_folder/test_data.csv', test_data);
% Crearea etichetelor pentru setul de antrenare
etichete_train = train_data(:, end);

% Etichetarea datelor: 1 pentru diabetic și 0 pentru sănătos
etichete_train(etichete_train == 2) = 0; % Sănătos
etichete_train(etichete_train == 4) = 1; % Diabetic

% Crearea etichetelor pentru setul de testare
etichete_test = test_data(:, end); 

% Etichetarea datelor: 1 pentru diabet și 0 pentru sănătos
etichete_test(etichete_test == 2) = 0; % Sănătos
etichete_test(etichete_test == 4) = 1; % Diabetic

% Parametri pentru metoda gradientului
m = 10; % Numărul de neuroni pe stratul ascuns
epsilon = 1e-3; % Criteriul de oprire pentru gradient

% Extinderea setului de date de antrenare cu o coloană de 1-uri
train_data_extended = [train_data, ones(train_size, 1)];

%% Metoda gradient
% Initializare parametri
X = randn(n+1, m); % +1 pentru stratul ascuns
x = rand(m, 1);

% variabile pentru stocare

vector_norma_gradient = [];
timp_gradient = [];
iter = 0;
maxiter = 1000;

% Iteratii metoda gradientului
t=tic;
while true
    y=zeros(train_size,1)
    % Calculul funcției obiectiv
    y = f_activare(train_data_extended * X) * x;
   loss = obiectiv(etichete_train, y);
  L_x = max(eig(x'*x));
  L_X = max(eig(X'*X));
  alpha=1/L_x;
  alpha1=1/L_X;
    % Calculul gradientului
    gradientul = gradientulll(etichete_train, y, X, train_data_extended);

  %  disp(gradientul);
    %break;
    gradient_norm = norm(gradientul);
    vector_norma_gradient = [vector_norma_gradient;gradient_norm];
   
    % Verificare criteriu de oprire
    if gradient_norm < epsilon || iter >=maxiter
        break;
    end

    % Actualizare parametrii
    x = x - alpha * gradientul';
     X = X - alpha1 * gradientul;
     timp_gradient=[timp_gradient,toc(t)]
    iter = iter + 1;
end


x_gradient = x;
X_gradient = X;
%semilogx(x_gradient);

% Afisarea rezultatelor metodei gradientului
% fprintf('Rezultate pentru metoda gradientului:\n');
% fprintf('Parametrii optimi x: %s\n', num2str(x_gradient'));
%fprintf('Timpul de execuție: %f secunde\n', timp_gradient);

% Afișare grafic evoluție norma gradientului
figure;
semilogy(vector_norma_gradient,'b','LineWidth',2);
title('Evoluție norma gradientului pentru Metoda Gradient');
xlabel('Iterație');
ylabel('Norma gradientului');

% Afișare grafic evoluție timp de execuție
figure;
plot(1:length(timp_gradient), timp_gradient);
title('Evoluție timp de execuție pentru Metoda Gradient');
xlabel('Iterație');
ylabel('Timp de execuție (secunde) MG');


% Interpolare pentru timp_gradient
timp_gradient_interp = linspace(min(timp_gradient), max(timp_gradient), length(vector_norma_gradient));
figure;
title('Evolutie in timp pentru Metoda Gradient');
semilogy(timp_gradient_interp, vector_norma_gradient, 'b', 'LineWidth', 2);
xlabel('Timp de execuție (secunde)');
ylabel('Norma gradientului');
grid on;


%%Metoda Newton
epsi = 1e-3;
X_n = randn(n+1, m); % Adăugăm un rând pentru termenul de bias
x_n = rand(m, 1);

% Initializare variabile pentru stocarea rezultatelor
pas_const = [];
timp_newton = [];
iter = 0;
maxiter = 1000;
alpha2=0.5;
% Iteratii metoda Newton
ti = tic;
while true
    y_n = f_activare(train_data_extended * X_n) * x_n;
    loss_n = obiectiv(etichete_train, y_n);

    % Calculul gradientului
    gradientul_n = gradientulll(etichete_train, y_n, X_n, train_data_extended);
    % Calculul hessienei
    hess_n = hessiana(etichete_train, y_n, X_n, train_data_extended);
    
    % Verificare criteriu de oprire
    if norm(gradientul_n) < epsi || iter >= maxiter
        break;
    end

    % Actualizare parametrii
    x_n=x_n -alpha2*(inv(hess_n))*(gradientul_n)';
    X_n=X_n -alpha2*(inv(hess_n))*(gradientul_n)';
    
    timp_newton = [timp_newton, toc(ti)];
    pas_const=[pas_const,norm(gradientul_n)];
    iter = iter + 1;
end

x_n_optim = x_n;
X_n_optim = X_n;

% Afișare grafic evoluție norma gradientului pentru metoda Newton
figure;
semilogy(pas_const,'b','LineWidth',2);
title('Evoluție norma gradientului  Metoda Newton');
xlabel('Iterație');
ylabel('Norma gradientului');

% Afișare grafic evoluție timp de execuție metoda Newton
figure;
plot(1:length(timp_newton), timp_newton);
title('Evoluție timp de execuție Metoda Newton');
xlabel('Iterație');
ylabel('Timp de execuție (secunde) Newton');

% Interpolare pentru timp_newton
timp_newton_interp = linspace(min(timp_newton), max(timp_newton), length(pas_const));
figure;
title('Evolutie in timp pentru Metoda Newton');
semilogy(timp_newton_interp, pas_const, 'b', 'LineWidth', 2);
xlabel('Timp de execuție (secunde)');
ylabel('Norma gradientului');
grid on;



% Extindem setul de date de testare cu o coloană de 1-uri
test_data_extended = [test_data, ones(size(test_data, 1), 1)];

% Calculul etichetelor prezise pentru setul de testare cu metoda Gradient
y_test_gradient = round(f_activare(test_data_extended * X_gradient) * x_gradient);

y_test_gradient(y_test_gradient >= 0.5) = 1;
y_test_gradient(y_test_gradient < 0.5) = 0;

% Calculul etichetelor prezise pentru setul de testare cu metoda Newton
y_test_newton = round(f_activare(test_data_extended * X_n_optim) * x_n_optim);

y_test_newton(y_test_newton >= 0.5) = 1;
y_test_newton(y_test_newton < 0.5) = 0;

% Matricea de confuzie pentru setul de testare pentru metoda Gradient
C_test_gradient = confusionmat(etichete_test, y_test_gradient);
disp('Matricea de confuzie pentru setul de testare:');
disp(C_test_gradient);

% Scorul F1 pentru setul de testare pentru metoda Gradient
F1_test_gradient = f1_score(etichete_test, y_test_gradient);
disp(['Scorul F1 pentru setul de testare : ', num2str(F1_test_gradient)]);

% Matricea de confuzie pentru setul de testare pentru metoda Newton
C_test_newton = confusionmat(etichete_test, y_test_newton);
disp('Matricea de confuzie pentru setul de testare pentru metoda Newton:');
disp(C_test_newton);

% Scorul F1 pentru setul de testare pentru metoda Newton
F1_test_newton = f1_score(etichete_test, y_test_newton);
disp(['Scorul F1 pentru setul de testare pentru metoda Newton: ', num2str(F1_test_newton)]);

