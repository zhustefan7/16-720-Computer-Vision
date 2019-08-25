
x = linspace(-20,20,100)
y = 1./(1+exp(-x))
deriv_y = (1+exp(-x)).^(-2).*exp(-x)
figure;
plot(x,y),hold on;
plot(x,deriv_y)
legend({'sigmoid' , 'derivative of sigmoid'})
title('sigmoid and derivative of sigmoid')


tanx = (1-exp(-2.*x))./(1+exp(-2.*x))
diff_tanx = 1 - tanx.^2
figure;
plot(x, tanx);
hold on;
plot(x , diff_tanx)
