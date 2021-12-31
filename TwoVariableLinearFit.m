%vectorizado y con solo valores calculados
clear ; close all; clc;
%Tengo que buscar forma de pasar como parámetro el nombre de los
%archivos para correr cada caso. Por ahora, solo comento y descomento
load "2-variable-dataset";
X = data(:,1:2);
Y = data(:,3);
%load house1Price;
%load house1Feat;
%load casas.dat;
%load precios.dat;
%s controla la cantidad de samples
s = length(Y);
X=[ones(size(X,1),1) X];%added bias column
alpha = 1.5;
%Feature scaling and mean normalization
%Vectorized
X(:,2:end)=(X(1:s,2:end)-mean(X(1:s,2:end)))./std(X(1:s,2:end));
theta0=zeros(size (X,2),1);

%Optimización usando fminunc
options=optimset('GradObj','on','MaxIter',400);
[theta Jobj] =fminunc(@(t)(costGradient(t,X,Y,0)),theta0,options);
disp "***************Entrenando parametros del estudio *********"
%disp "Valor final de theta por gradient descent:", theta
disp "Valor optimo del objetivo por gradient descent:"; Jobj
%theta = pinv(X'*X)*X'*Y;
%Jobj=sum((X*theta-Y).^2)/length(Y);
%disp "Valor final de theta por ecuación normal:", theta
%disp "Valor optimo del objetivo por ecuacion normal"; Jobj
´%Ejecuté este caso en python con el siguiente código:
% from sklearn.linear_model import LinearRegression
% df = pd.read_csv('https://bit.ly/2X1HWH7', delimiter=",")
% X = df.values[:, :-1]
% Y = df.values[:, -1]
% fit = LinearRegression().fit(X, Y)
% print("z = {0} + {1}x + {2}y".format(fit.intercept_, fit.coef_[0], fit.coef_[1]))
%% se imprime esto:
% z = 20.109432820035977 + 2.0067264725128062x + 3.002037976646692y
% fit.coef_[0]*X[:,0]+fit.coef_[1]*X[:,1]+fit.intercept_
% lo anterior imprime las predicciones (!?) para cada valor de X con el arrelgo de ceficientes ya entrenadoa
% eso da exactamente igual que la ejecuci´no en Octave, por ejemplo:
% 86.15426831,  61.14265299,  67.14672894,  64.14469097,
% 78.15956934,  63.14937946,  57.14530351,  32.13368819,
% 
% Eso ocurre a la perfección, aún cuando en Octave hice mean normalization y feature scaling y en python no
%
%
%