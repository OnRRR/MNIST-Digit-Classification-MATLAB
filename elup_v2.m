function fr = elup_v2(x)
    %f = zeros(length(x),1);
    f = zeros(size(x));
    for i = 1:size(x,1)
        for k = 1:size(x,2)
            if x(i,k)>=0
                f(i,k) = 1;
            else
                f(i,k) = 0.2*exp(x(i,k));
            end
        end
    end
    fr = f;
end