
#===============================
#   Codigos IA
# Gradientes Simples Con Pytorch
#===============================
#   Luna Miranda Hershel Brandon
#   ESFM-IPN
#==============================


import torch

x = torch.tensor(1.0)
y = torch.tensor(2.0)

#===========================================
# Variable de diferenciación (d/dw) -> requires_grad=True
#===========================================
w = torch.tensor(1.0, requires_grad=True)

#===========================================
# evaluación cálculo de costo
#===========================================
y_predicted = w * x
loss = (y_predicted - y)**2
print(loss)

#===========================================
# retropropagación para calcular gradiente
# w.grad es el gradiente
#===========================================
loss.backward()
print(w.grad)

#===========================================
# Nuevos coeficientes (descenso de gradiente)
# repetir evaluación y retropropagación
#===========================================
with torch.no_grad():
    w -= 0.01 * w.grad
w.grad.zero_()
print(w)