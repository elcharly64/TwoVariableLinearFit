%Función para calcular el objetivo y gradiente
function [Jobj, Grad] = costGradient(Theta,X,Y,lambda)

[m n] = size(X);
%n ya incluye el bias

Jobj=sum((X*Theta-Y).^2/2/m);
Grad=X'*(X*Theta-Y)/m;



end