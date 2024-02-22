function fr = elu_v2(x)
    %f = zeros(length(x),1);
    f = zeros(size(x));
    for i = 1:size(x,1)
        for k = 1:size(x,2)
            if x(i,k)>=0
                f(i,k) = x(i,k);
            else
                f(i,k) = 0.2*(exp(x(i,k))-1);
            end
        end
    end
    fr = f;
end