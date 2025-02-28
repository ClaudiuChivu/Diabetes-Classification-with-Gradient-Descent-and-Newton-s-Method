function f1 = f1_score(y_true, y_pred)
    C = confusionmat(y_true, y_pred);
    precizie = C(2,2) / (C(2,2) + C(1,2));
    sensibilitate = C(2,2) / (C(2,2) + C(2,1));
    f1 = 2 * (precizie * sensibilitate) / (precizie + sensibilitate);
end
