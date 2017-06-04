function [h] = plotPeelingDemo(Data, all, id)


h = figure('Color', [1 1 1]);
if all==1
    clear X_a X_p X_o
    
    X_a(1:13,:) = Data{id}.active.X;
    X_p(1:13,:) = Data{id}.passive.X;
    X_o(1:6,:)  = Data{id}.object.feats;
    
    
    subplot(4,1,1)
    plot(X_a(1:3,:)')
    legend({'x','y','z'},'Interpreter','LaTex')
    title('Active arm', 'Interpreter','LaTex')
    xlim([1 length(X_a)])
    grid on
    
    subplot(4,1,2)
    plot(X_a(8:10,:)')
    legend({'$f_x$','$f_y$','$f_z$'},'Interpreter','LaTex')
    title('Active arm', 'Interpreter','LaTex')
    xlim([1 length(X_a)])
    grid on
    
    subplot(4,1,3)
    plot(X_p(4:7,:)')
    legend({'$q_i$','$q_j$','$q_k$','$q_w$'},'Interpreter','LaTex')
    title('Passive arm', 'Interpreter','LaTex')
    xlim([1 length(X_p)])
    grid on
    
    subplot(4,1,4)
    x_o = X_o(1:6,:);
    % Rate of change of color
    x_o_dot = [zeros(6,1) diff(x_o')'];
    % Smoothed out with savitksy golay filter
    x_o_dot = sgolayfilt(x_o_dot', 6, 151)';
        
    plot(x_o_dot(1:end,:)')
    legend({'$\dot{\mu}_r$','$\dot{\mu}_g$','$\dot{\mu}_b$','$\dot{\sigma}_r$','$\dot{\sigma}_g$','$\dot{\sigma}_b$'}, 'Interpreter','LaTex')
    title('Object Features', 'Interpreter','LaTex')
    xlim([1 length(X_o)])
    grid on
   
else
    
    clear X_a X_p 
    X_a(1:13,:) = Data{id}.active.X;
    X_p(1:13,:) = Data{id}.passive.X;
        
    subplot(3,1,1)
    plot(X_a(1:3,:)')
    legend({'x','y','z'},'Interpreter','LaTex')
    title('Active arm', 'Interpreter','LaTex')
    xlim([1 length(X_a)])
    grid on
    
    subplot(3,1,2)
    plot(X_a(8:10,:)')
   legend({'$f_x$','$f_y$','$f_z$'},'Interpreter','LaTex')
    title('Active arm', 'Interpreter','LaTex')
    xlim([1 length(X_a)])
    grid on
    
    subplot(3,1,3)
    plot(X_p(4:7,:)')
    legend({'$q_i$','$q_j$','$q_k$','$q_w$'},'Interpreter','LaTex')
    title('Passive arm', 'Interpreter','LaTex')
    xlim([1 length(X_p)])
    grid on
    
    
end


end