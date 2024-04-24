function [X,y,tX,ty] = randomData(m,r)

% This file aims at generating random data  
% Inputs:
%       m     -- number of samples
%       r     -- flipping ratio
% Outputs:
%       X     --  training samples data,    m/2-by-n
%       Y     --  training samples classes, n-by-1
%       tX    --  testing  samples data,    m/2-by-n
%       ty    --  testing  samples classes, n-by-1
%
% written by Shenglong Zhou, 10/05/2020
m2    = ceil(m/2);
rng('shuffle');

A   = [mvnrnd([0.5;-3],[0.2 0;0 3],m2);
       mvnrnd([-0.5;3],[0.2 0;0 3],m2)];
c   = [-ones(m2,1); ones(m2,1)];    
T   = randperm(m); 
X   = A(T(1:m2),:); 
y   = c(T(1:m2)); 
y   = filp(y,r);  

tX = A(T(m2+1:m),:); 
ty = c(T(1+m2:m));
ty   = filp(ty,r);
clear A c T q
end

function fc = filp(fc,r)
      if r  > 0
         mc = length(fc) ;    
         T0 = randperm(mc);  
         fc(T0(1:ceil(r*mc)))=-fc(T0(1:ceil(r*mc)));
     end
end