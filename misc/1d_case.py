import numpy as np
import matplotlib.pyplot as plt
import time


f0 = 300e3
dr = 1e-6
dt = 1e-10
t_final = 100e-6
a0 = 5e-6
r = a0 + np.arange(50)*dr

eta = 1.2e-3

ur = np.zeros((len(r)))

p = np.zeros((len(r)-2))
psi = p.copy()

t = 0
i = 0

idx = 0

while t < t_final:

    ur[0] = 1e-6*np.sin(2*np.pi*f0*t)

    u_drr = (ur[2:] - 2 *ur[1:-1] + ur[:-2])/(dr**2)

    u_dr_r = (ur[2:] - ur[:-2])/(2*dr*r[1:-1])

    u_r2 = ur[1:-1]/(r[1:-1]**2)

    urn  = ur[1:-1] + dt*eta*(u_drr + u_dr_r - u_r2)

    t+=dt
    ur[1:-1] = urn

    p += urn

    #psi[i] = urn[idx]*dt + a0
    #idx = (np.abs(psi[i]-r)).argmin()


    i+=1



p_new = p / i

plt.plot(r[1:-1],urn)

plt.show()

time.sleep(10)


"""


 
%for n=1:10,

 
t=0;tout=0;
while t<tfinal,

 
urn=0*r;

 
urr=(ur(3:end)-2*ur(2:end-1)+ur(1:end-2))/dr^2;

 
rur=(ur(3:end)-ur(1:end-2))./(2*r(2:end-1)*dr);

 
urr2=ur./r.^2;

 
% figure(1);
% plot(r(2:end-1),urr,r(2:end-1),rur,r,urr2);

 
urn(2:end-1)=ur(2:end-1) + deltat*nu*(urr+rur-urr2(2:end-1));

 
%urn(1)=(4/3)*ur(2)-(1/3)*ur(3);

 
ur=urn;
t=t+deltat;
ur(1)=1e-6*sin(2*pi*f0*t);

 
if t>tout,
    figure(2)
plot(r,ur);
tout=tout+1e-6;
hold on;
pause(0.1);
end

 
end
hold off;

 
% 
% figure(2);
% plot(r,ur);pause(1);

 
%urr=ur(2:end,:)-ur(

"""