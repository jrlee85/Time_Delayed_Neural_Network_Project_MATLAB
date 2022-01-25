%% Original Batch Learning Code

% This is the original batch learning code provided in the coursework
% brief. It is included here in order to provide a comparison of the
% results.

% Set the random seed to default in order to compare batch learning
% version, matrix incremental version, and looped incremental version
rng('default')

% test_dnn.m
%clear all;

T  = 500; Nu = 100; b = 0.1; c = 0.2; tau = 17;
% Initial values obtained by a random function
xi = [ 0.401310 0.953833 0.174821 0.572708 0.971513 0.109872 0.388265 ...
      0.942936 0.213617 0.666899 0.881914 0.413442 0.962755 0.142354 ...
      0.484694 0.991570 0.033185 0.127373 0.441263 0.978804]'; 
% Mackay-Glass time series generation
for t = 20:T+49
  xi(t+1) = xi(t)+c*xi(t-tau)/(1+xi(t-tau).^10)-b*xi(t);
end
xi(1:50) = [];
%--------------------------------------------------------------------------
relNums=xi(100:T,1); %ynrmv=mean(relNums(:)); sigy=std(relNums(:)); 
nrmY=relNums; %nrmY=(relNums(:)-ynrmv)./sigy; 
ymin=min(nrmY(:)); ymax=max(nrmY(:)); 
relNums=2.0*((nrmY-ymin)/(ymax-ymin)-0.5);
% create a matrix of lagged values for a time series vector
Ss=relNums';
idim=10; % input dimension
odim=length(Ss)-idim; % output dimension
for i=1:odim
   y(i)=Ss(i+idim);
   for j=1:idim
       x(i,j) = Ss(i-j+idim); %x(i,idim-j+1) = Ss(i-j+idim);
   end
end
examples = x'; targets = y; NHID = 5; prnout = targets;
[NINP,NPATS] = size(examples); [NOUT,NP] = size(targets);
%
eta = 0.0001; L = 0.02; inputs = [examples;ones(1,NPATS)]; sig = ones(1,NP); 
w1 = 0.5*(rand(NHID,1+NINP)-0.5); w1s = 0.5*(rand(NHID,1+NINP)-0.5);
w2 = 0.5*(rand(1,1+NHID)-0.5); w2s = 0.5*(rand(1,1+NHID)-0.5); 
% Phase 1
for epoch = 1:200
  % Forward propagation (mean network):
  sum1 = w1*inputs; hidden = tanh(sum1);
  sum2 = w2*[hidden;ones(1,NPATS)]; out = sum2; 
  % Backpropagation (mean network):
  error = targets-out; sse = sum(sum(error.^2)); 
  bout = error./sig; 
  bp = (w2'*bout);
  bh = (1.0-hidden.^2).*bp(1:end-1,:);
  % Computing weight deltas (mean network):
  dW2 = bout*[hidden;ones(1,NPATS)]'; 
  dW1 = bh*inputs';
  % Updating the weights (mean network):
  w2 = w2+eta*dW2; 
  w1 = w1+eta*dW1;
end
% Phase 2
etas = 0.00001;
for epoch = 1:200
  % Forward propagation (variance network):
  sum1s = w1s*inputs; hiddens = tanh(sum1s);
  sum2s = w2s*[hiddens;ones(1,NPATS)]; sig = exp(sum2s); 
  % Backpropagation (variance network): 
  bouts = ((error.*error)./sig-1.0)/2;
  bps = (w2s'*bouts);
  bhs = (1.0-hiddens.^2).*bps(1:end-1,:);
  % Computing weight deltas (variance network):
  dW2s = bouts*[hiddens;ones(1,NPATS)]'; 
  dW1s = bhs*inputs';
  % Updating the weights (variance network):
  w2s = w2s+etas*dW2s; 
  w1s = w1s+etas*dW1s;
end
% Phase 3
for epoch = 1:500
  sum1 = w1*inputs; hidden = tanh(sum1);
  sum2 = w2*[hidden;ones(1,NPATS)]; out = sum2; 
  error = targets-out; sse = sum(sum(error.^2)); if (sse<L), break, end
  bout = error./sig; 
  bp = (w2'*bout);
  bh = (1.0-hidden.^2).*bp(1:end-1,:);
  dW2 = bout*[hidden;ones(1,NPATS)]'; 
  dW1 = bh*inputs';
  w2 = w2+eta*dW2; 
  w1 = w1+eta*dW1;
  sum1s = w1s*inputs; hiddens = tanh(sum1s);
  sum2s = w2s*[hiddens;ones(1,NPATS)]; sig = exp(sum2s); outs = sig;
  bouts = ((error.*error)./sig-1.0)/2;
  bps = (w2s'*bouts);
  bhs = (1.0-hiddens.^2).*bps(1:end-1,:);
  dW2s = bouts*[hiddens;ones(1,NPATS)]'; 
  dW1s = bhs*inputs';
  w2s = w2s+etas*dW2s; 
  w1s = w1s+etas*dW1s;
end
% Display the number or epochs (to see if early stopping has occurred)
fprintf('No. epochs run (batch version): %f\n', epoch);
% Display the final error in order to compare models
fprintf('Final error (batch version): %f\n', sse);
prnout = out; conf1 = prnout+1.645*(outs); conf2 = prnout-1.645*(outs);
plot([81:300],targets(81:300),'b',[81:300],prnout(81:300),'r',...
     [81:300],conf1(81:300),'g',[81:300],conf2(81:300),'g')
title('Mackay-Glass time series')

%% Clear workspace

clear
%% Matrix Incremental Version

% My work begins here.

% Set the random seed to default in order to compare batch learning version,
% matrix incremental version, and looped incremental version
rng('default')

% Code below lifted directly from coursework brief
% Here we generate artificial data using the Mackey-Glass equation. T is the
% number of inputs, Nu is not actually used in this case, and b, c, tau and
% xi are used in conjunction with the equation to generate the values.
T = 500; Nu = 100; b = 0.1; c = 0.2; tau = 17;
xi = [ 0.401310 0.953833 0.174821 0.572708 0.971513 0.109872 0.388265...
       0.942936 0.213617 0.666899 0.881914 0.413442 0.962755 0.142354...
       0.484694 0.991570 0.033185 0.127373 0.441263 0.978804]';
% xi is filled with values from the Mackey-Glass equation (up to position
% 549) before the first 50 values are dropped  
for t = 20:T+49
    xi(t+1) = xi(t)+c*xi(t-tau)/(1+xi(t-tau).^10)-b*xi(t);
end
xi(1:50) = [];
%--------------------------------------------------------------------------
% relNums is a 401x1 vector of values from xi from position 100 to 500
relNums=xi(100:T,1); 
nrmY=relNums;
% We then normalise the values in relNums/nrmY using min-max normalisation,
% and by subtracting 0.5 and multiplying by 2 to get values between -1 and
% 1.
ymin=min(nrmY(:)); ymax=max(nrmY(:));
relNums=2.0*((nrmY-ymin)/(ymax-ymin)-0.5);
% Ss (1x401 vector) is set to the transpose of the normalised values
Ss=relNums';
idim=10;
odim=length(Ss)-idim; 
% Create a vector (y) and matrix (x) to hold our targets and inputs
% respectively. y is a 1x391 vector and x is a 391x10 matrix, meaning each
% single example consists of 10 input values, and we are using 391 examples
% in total.
for i=1:odim
    y(i)=Ss(i+idim);
    for j=1:idim
        x(i,j) = Ss(i-j+idim); 
       end
end
% Set examples to be the transpose of x, targets to be the vector y, NHID 
% (number of hidden nodes) to be 5, prnout to be the targets (y), NINP
% (number of inputs) to be the row size of % examples (10), NPATS
% (number of patterns) to be the column size of % examples (391),
% NOUT (number of outputs) to be the row size of targets % (1) and  NP 
% (number of output patterns) to be the column size of targets (391).
examples = x'; targets = y; NHID = 5; prnout = targets;
[NINP,NPATS] = size(examples); [NOUT,NP] = size(targets);
% Phase 1
% Our neural network is made up of two sections, one which learns the mean 
% and one which learns the variance. Phase 1 learns the mean, which is a 
% prediction of the next value in the series.
% eta is the learning rate used when updating the weights. L is the
% training threshold - if the sum of squared errors in phase 3 falls below
% this value, training will stop early.
% inputs is the inputs that will be fed into the network. It consists of
% the examples matrix created above, plus a value of 1 for each column, 
% resulting in a 11x391 matrix. This 1 represents the bias.
% sig is initialised as a 1x391 vector of 1s. This will be used as the
% output of the variance section of the network.
% w1 is a 5x11 matrix of weights for the input to hidden nodes for phase 1.
% Initial values are generated between 0 and 1, and then set to values
% between -0.25 and +0.25 (as we don't want the weights to be too large or
% too small initially). w1s is similarly a 5x11 matrix of initialised
% weights for the input to hidden nodes, this time for phase 2 (the
% variance phase). 5 represents the number of hidden nodes, and 11
% represents the number of inputs plus 1 bias term.
% w2 and w2s are 1x6 vectors representing the weights from hidden to output
% for phase 1 and phase 2 respectively. 1 represents the number of output
% nodes and 6 represents the number of hidden nodes plus 1 bias term.
eta = 0.001; L = 0.02; inputs = [examples;ones(1,NPATS)]; sig = ones(1,NP);
w1 = 0.5*(rand(NHID,1+NINP)-0.5); w1s = 0.5*(rand(NHID,1+NINP)-0.5);
w2 = 0.5*(rand(1,1+NHID)-0.5); w2s = 0.5*(rand(1,1+NHID)-0.5);
% We now train the mean section of the network for 200 epochs.
for epoch = 1:200
    % First we do the forward propagation, setting sum1 to be the input
    % to hidden weights multiplied by the inputs, using the tanh
    % activation function on sum1 at the hidden layer (hidden), and then
    % calculating the final output as the hidden to output weights 
    % multiplied by hidden (with appended ones). 
    sum1 = w1*inputs; hidden = tanh(sum1);
    sum2 = w2*[hidden;ones(1,NPATS)]; out = sum2;
    % To perform the backpropagation, we calculate the error by subtracting
    % the outputs of the network from the targets (correct values), before
    % calculating the sum of squared errors. 
    error = targets-out; sse = sum(sum(error.^2));
    % We set bout to error
    bout = error;
    % Multiplying bout by the transpose of the weights from hidden to output
    % gives the back propagated error (bp). We then multiply this by the
    % derivative of the activation function at the hidden layer (bh). 
    bp = (w2'*bout);
    bh = (1.0-hidden.^2).*bp(1:end-1,:);
    % To get the gradient for updating the hidden to output weights, we
    % multiply the error (bout) by the output of the hidden layer (hidden)
    % appended with ones. 
    dW2 = bout*[hidden;ones(1,NPATS)]';
    % To get the gradient for updating the input to hidden weights, we
    % multiply bh by the input values.
    dW1 = bh*inputs';
    % Finally, we update the weights using the gradients and the learning
    % rates defined earlier.
    w2 = w2+eta*dW2;
    w1 = w1+eta*dW1;
end
% Phase 2
% Phase 2 calculates the variance, which gives and indication of how
% confident we are in the prediction of the mean section of the network
% (phase 1). Again, we set a learning rate for the weight updates, and run
% the training for 200 epochs.
etas = 0.00001;
for epoch = 1:200
  % This is the forward propagation step, which is almost identical to
  % phase 1. The only difference here is that the output node outputs the
  % exponent of sum2s. This is assigned to the variable sig.
  sum1s = w1s*inputs; hiddens = tanh(sum1s);
  sum2s = w2s*[hiddens;ones(1,NPATS)]; sig = exp(sum2s);
  % bouts here is the square of the error divided by the output of the
  % network minus 1, divided by 2.
  bouts = ((error.*error)./sig-1.0)/2;
  % The remaining values are calculated in the same way as for phase 1.
  bps = (w2s'*bouts);
  bhs = (1.0-hiddens.^2).*bps(1:end-1,:);
  dW2s = bouts*[hiddens;ones(1,NPATS)]'; 
  dW1s = bhs*inputs';
  w2s = w2s+etas*dW2s; 
  w1s = w1s+etas*dW1s;
end
% Phase 3
% Phase 3 trains both sections of the network together. This section of
% code demonstrates the matrix incremental version.
% We set new learning rates for phase 3. I found the best results were
% obtained by using separate learning rates for phase 1&2 versus phase 3.
% After trying several learning rates, I found these values to give the
% lowest error.
eta_p3 = 0.000013;
etas_p3 = 0.00014;
% Because we are training the network incrementally, we initialise vectors 
% to hold the values of out, outs and error after each loop
final_out = zeros(1,391);
final_outs = zeros(1,391);
final_error = zeros(1,391);
% We will also keep track of the SSE for each epoch in order to plot a loss
% curve. We initialise a vector with zeros.
ongoing_sse = zeros(500,1);
% We run the training in phase 3 for 500 epochs
for epoch = 1:500
    % Instead of using all of the inputs examples at once, this time we loop
    % through each example one at a time
    for e = 1:NPATS
        % Use the e-th input to calculate sum1, hidden, sum2 and out (in 
        % the same manner as phase 1)
        sum1 = w1*inputs(:,e); hidden = tanh(sum1);
        sum2 = w2*[hidden;ones(1,1)]; out = sum2;
        % Use the e-th traget to calculate the error
        error = targets(e)-out;
        % In phase 3, bout is caluclated as the error divided by the e-th
        % value of sig (the output of phase 2)
        bout = error./sig(e);
        % bp, bh, dW2 and dW1 are all calculated as per phase 3(expect here
        % only using one input example)
        bp = (w2'*bout);
        bh = (1.0-hidden.^2).*bp(1:end-1,:);
        dW2 = bout*[hidden;ones(1,1)]';
        dW1 = bh*inputs(:,e)';
        % Use the phase 3 learning rates to calculate the new weights
        w2 = w2+eta_p3*dW2;
        w1 = w1+eta_p3*dW1;
        % We then move directly on to the second section of the network 
        % during the same epoch. Again, the forward propagation and 
        % backward propagation are performed in essentially the same way as
        % explained above in phase 2, except that here we are only using 
        % one input example.
        sum1s = w1s*inputs(:,e); hiddens = tanh(sum1s);
        % We set outs to be the output of the variance section of the
        % network.
        sum2s = w2s*[hiddens;ones(1,1)]; sig(e) = exp(sum2s); outs=sig(e);
        % Again, the remaining variables are calculated in the same way as
        % phase 2 expect using only one input example. The new learning
        % rates are used to update the weights.
        bouts = ((error.*error)./sig(e)-1.0)/2;
        bps = (w2s'*bouts);
        bhs = (1.0-hiddens.^2).*bps(1:end-1,:);
        dW2s = bouts*[hiddens;ones(1,1)]';
        dW1s = bhs*inputs(:,e)';
        w2s = w2s+etas_p3*dW2s;
        w1s = w1s+etas_p3*dW1s;
        % Finally, we update the values of final_out using the out from
        % the mean section of the network. These are our predictions.
        % final_outs is updated with the output (outs) of the variance
        % section of the network. final_error is updated using the error
        % calculated in the mean section of the network.
        final_out(e) = out;
        final_outs(e) = outs;
        final_error(e) = error;  
    end
    % After passing each input example through the network, we calculate
    % the final sum of squares error using the final_error vector. This
    % gets updated for each epoch. If the SSE is less than the threshold at
    % the end of any one epoch, training is stopped early.
    sse = sum(sum(final_error.^2)); if (sse<L), break, end
    % We also update the ongoing_sse vector with the SSE from that epoch
    ongoing_sse(epoch) = sse;
end
% We display the number or epochs in order to see if early stopping has
% occurred
fprintf('No. epochs run (matrix version): %f\n', epoch);
% We display the final error in order to compare models
fprintf('Final error (matrix version): %f\n', sse);
% We can plot the loss curve using our ongoing_sse vector to see how 
% the loss reduces as training goes on
plot(ongoing_sse)
% Add a title and labels to the plot
title('Loss curve')
xlabel('epoch')
ylabel('SSE')
% Next, we plot the predictions (red) along with the targets (correct 
% values, blue) for the 81st to 300th values. We also plot the 95%
% confidence intervals (z = 1.645, green and yellow, dashed lines) using 
% the variance % values calculated by the variance section of the network.
prnout = final_out; conf1 = prnout+1.645*(final_outs); 
conf2 = prnout-1.645*(final_outs);
figure; plot([81:300],targets(81:300),'b',[81:300],prnout(81:300),'r',...
    [81:300],conf1(81:300),'g--',[81:300],conf2(81:300),'y--')
% A title and legend are added to the plot
title('Mackay-Glass time series')
legend('Actual', 'Predictions', 'Upper conf interval', 'Lower conf interval')

%% Clear workspace

clear
%% Loop Incremental Version

% This section implements the incremental training version of the code, but
% instead of using matrix multiplication in phase 3, it uses for loops.

% For a full explanation of each step in the code, please see the matrix
% incremental version above this section.

% Again, we se the random seed to default in order to compare versions.
rng('default')

% As with the matreix version above, the code below is taken directly from
% the coursework brief. My code begins at phase 3.
T = 500; Nu = 100; b = 0.1; c = 0.2; tau = 17;
xi = [ 0.401310 0.953833 0.174821 0.572708 0.971513 0.109872 0.388265...
       0.942936 0.213617 0.666899 0.881914 0.413442 0.962755 0.142354...
       0.484694 0.991570 0.033185 0.127373 0.441263 0.978804]';
for t = 20:T+49
    xi(t+1) = xi(t)+c*xi(t-tau)/(1+xi(t-tau).^10)-b*xi(t);
end
xi(1:50) = [];

relNums=xi(100:T,1); 
nrmY=relNums;
ymin=min(nrmY(:)); ymax=max(nrmY(:));
relNums=2.0*((nrmY-ymin)/(ymax-ymin)-0.5);

Ss=relNums';
idim=10;
odim=length(Ss)-idim; 
for i=1:odim
    y(i)=Ss(i+idim);
    for j=1:idim
        x(i,j) = Ss(i-j+idim); 
       end
end
examples = x'; targets = y; NHID = 5; prnout = targets;
[NINP,NPATS] = size(examples); [NOUT,NP] = size(targets);
% Phase 1
eta = 0.001; L = 0.02; inputs = [examples;ones(1,NPATS)]; sig = ones(1,NP);
w1 = 0.5*(rand(NHID,1+NINP)-0.5); w1s = 0.5*(rand(NHID,1+NINP)-0.5);
w2 = 0.5*(rand(1,1+NHID)-0.5); w2s = 0.5*(rand(1,1+NHID)-0.5);

for epoch = 1:200
    sum1 = w1*inputs; hidden = tanh(sum1);
    sum2 = w2*[hidden;ones(1,NPATS)]; out = sum2;
    error = targets-out; sse = sum(sum(error.^2));
    bout = error./sig;
    bp = (w2'*bout);
    bh = (1.0-hidden.^2).*bp(1:end-1,:);
    dW2 = bout*[hidden;ones(1,NPATS)]';
    dW1 = bh*inputs';
    w2 = w2+eta*dW2;
    w1 = w1+eta*dW1;
end
% Phase 2
etas = 0.00001;
for epoch = 1:200
  sum1s = w1s*inputs; hiddens = tanh(sum1s);
  sum2s = w2s*[hiddens;ones(1,NPATS)]; sig = exp(sum2s); 
  bouts = ((error.*error)./sig-1.0)/2;
  bps = (w2s'*bouts);
  bhs = (1.0-hiddens.^2).*bps(1:end-1,:);
  dW2s = bouts*[hiddens;ones(1,NPATS)]'; 
  dW1s = bhs*inputs';
  w2s = w2s+etas*dW2s; 
  w1s = w1s+etas*dW1s;
end

% Phase 3
% As above, separate learning rates are used for phase 3 and vectors are
% initialised for the final values
eta_p3 = 0.000013;
etas_p3 = 0.00014;
final_out = zeros(1,391);
final_outs = zeros(1,391);
final_error = zeros(1,391);
ongoing_sse = zeros(500,1);
for epoch = 1:500
    % Loop through each input
    for e = 1:NPATS
        % Set input to equal the e-th input
        input = inputs(:,e);
        % Because we are using for loops rather than matrix multiplication,
        % we must first initialise the vector. We initialise the vector with
        % zeros. These values will be updated in the for loop.
        sum1 = zeros(NHID,1);
        % We now loop through the values in each row of the weights matrix,
        % multiplying by the corresponding element in the column of the
        % input vector and summing the resultant values to get the correct
        % value for each row of the sum1 vector
        for i=1:height(sum1)
            for j=1:width(sum1)
                for k=1:height(input)
                    sum1(i,j) = sum1(i,j)+w1(i,k)*input(k,j);
                end 
            end 
        end
        hidden = tanh(sum1);
        % Initialise sum2 with 0
        sum2 = zeros(1,1);
        % Create hidden_extended vector, consisting of hidden plus a row
        % of 1
        hidden_extended = [hidden;ones(1,1)];
        % As above, loop through values of w2 and hidden_extended,
        % multiplying and summing, to get value for sum2
        for i=1:height(sum2)
            for j=1:width(sum2)
                for k=1:height(hidden_extended)
                    sum2(i,j) = sum2(i,j)+w2(i,k)*hidden_extended(k,j);
                end 
            end 
        end
        out = sum2;
        error = targets(e)-out;
        bout = error./sig(e);
        % Initialise bp
        bp = zeros(NHID+1,1);
        % Create new variable for w2 transpose
        w2t = w2';
        % Perform matrix multiplication with for loop
        for i=1:height(bp)
            for j=1:width(bp)
                for k=1:height(bout)
                    bp(i,j) = bp(i,j)+w2t(i,k)*bout(k,j);
                end 
            end 
        end
        bh = (1.0-hidden.^2).*bp(1:end-1,:);
        % Initialise dW2 and create hidden_extended_t from tranposes of
        % hidden with appended 1
        dW2 = zeros(1,NHID+1);
        hidden_extended_t = [hidden;ones(1,1)]';
        % Perform matrix multiplication with for loop
        for i=1:height(dW2)
            for j=1:width(dW2)
                for k=1:height(hidden_extended_t)
                    dW2(i,j) = dW2(i,j)+bout(i,k)*hidden_extended_t(k,j);
                end
            end
        end
        % Initalise dW1 and create input_t from transpose of
        % input
        input_t=input';
        dW1 = zeros(NHID,NINP+1);
        % Perform matrix multiplication with for loop
        for i=1:height(dW1)
            for j=1:width(dW1)
                for k=1:height(input_t)
                    dW1(i,j) = dW1(i,j)+bh(i,k)*input_t(k,j);
                end
            end
        end
        % Use the phase 3 learning rates to calculate the new weights
        w2 = w2+eta_p3*dW2;
        w1 = w1+eta_p3*dW1;
        % Initialise sum1s
        sum1s = zeros(NHID,1);
        % Perform matrix multiplication with for loop
        for i=1:height(sum1s)
            for j=1:width(sum1s)
                for k=1:height(input)
                    sum1s(i,j) = sum1s(i,j)+w1s(i,k)*input(k,j);
                end 
            end 
        end
        hiddens = tanh(sum1s);
        % Initialise sum2s with zero and create hidden_extended vector as
        % above
        sum2s = zeros(1,1);
        hiddens_extended = [hiddens;ones(1,1)];
        % Perform matrix multiplication with for loop
        for i=1:height(sum2s)
            for j=1:width(sum2s)
                for k=1:height(hiddens_extended)
                    sum2s(i,j) = sum2s(i,j)+w2s(i,k)*hiddens_extended(k,j);
                end 
            end 
        end
        sig(e) = exp(sum2s); outs = sig(e);
        bouts = ((error.*error)./sig(e)-1.0)/2;
        % Initialise bps with zeros and create new variable for w2s
        % transpose as above
        bps = zeros(NHID+1,1);
        w2st = w2s';
        % Perform matrix multiplication with for loop
        for i=1:height(bps)
            for j=1:width(bps)
                for k=1:height(bouts)
                    bps(i,j) = bps(i,j)+w2st(i,k)*bouts(k,j);
                end 
            end 
        end        
        bhs = (1.0-hiddens.^2).*bps(1:end-1,:);
        % Initalise dW2s and create hiddens_extended_t from tranposes of
        % hiddens with appended 1
        dW2s = zeros(1,NHID+1);
        hiddens_extended_t = [hiddens;ones(1,1)]';
        % Perform matrix multiplication with for loop
        for i=1:height(dW2s)
            for j=1:width(dW2s)
                for k=1:height(hiddens_extended_t)
                    dW2s(i,j) = dW2s(i,j)+bouts(i,k)*hiddens_extended_t(k,j);
                end
            end
        end
        % Initialise dW1s and create input_t from transpose of
        % input
        input_t=input';
        dW1s = zeros(NHID,NINP+1);
        % Perform matrix multiplication with for loop
        for i=1:height(dW1s)
            for j=1:width(dW1s)
                for k=1:height(input_t)
                    dW1s(i,j) = dW1s(i,j)+bhs(i,k)*input_t(k,j);
                end
            end
        end
        % Use the phase 3 learning rates to calculate the new weights
        w2s = w2s+etas_p3*dW2s;
        w1s = w1s+etas_p3*dW1s;
        % Assign the value for out as the e-th value in the final_out
        % vector. Repeat for final_outs and final_error
        final_out(e) = out;
        final_outs(e) = outs;
        final_error(e) = error;
    end
    % Calculate the final sse from the final_error vector and update the
    % ongoing_see vector
    sse = sum(sum(final_error.^2)); if (sse<L), break, end
    ongoing_sse(epoch) = sse;
end
% Display the number or epochs (to see if early stopping has occurred)
fprintf('No. epochs run (loop version): %f\n', epoch);
% Display the final error in order to compare models
fprintf('Final error (loop version): %f\n', sse);
% As can be seen from the output display, the full 500 epochs are run for
% all three implementations. However, the overall performance of the 
% incremental versions is better than the batch training version (also note
% - the results for the matrix incremental implementation and the looped
% incremental implementation are the same, as expected).
% With the original learning rates for the batch training version,
% after 500 epochs of phase three, the SSE = 1.484471. With those same
% learning rates for the incremental version, the SSE is 0.791047. With the
% phase three learning rates set as above, the batch version has
% SSE = 1.031847. The incremental version has SSE = 0.054371 (see output).
% We do see early stopping if we change the threshold to 0.1. In this case,
% the incremental version stops after 302 epochs. The batch version does
% not stop early. Clearly, the incremental version is performing better
% than the batch version. However, it is slightly slower to run, especially
% the looped version.
% Graphs for each implementation can be viewed by running the separate
% sections above. They are also included as jpeg files in the zip file
% submission.
plot(ongoing_sse)
title('Loss curve')
xlabel('epoch')
ylabel('SSE')
prnout = final_out; conf1 = prnout+1.645*(final_outs); 
conf2 = prnout-1.645*(final_outs);
figure; plot([81:300],targets(81:300),'b',[81:300],prnout(81:300),'r',...
    [81:300],conf1(81:300),'g--',[81:300],conf2(81:300),'y--')
title('Mackay-Glass time series')
legend('Actual', 'Predictions', 'Upper conf interval', 'Lower conf interval')