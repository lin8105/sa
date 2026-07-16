function [FSIM, FSIMc] = FSIM(imageRef, imageDis)
    % 主函数开始
    [rows, cols] = size(imageRef(:,:,1));
    I1 = ones(rows, cols); I2 = ones(rows, cols);
    Q1 = ones(rows, cols); Q2 = ones(rows, cols);

    if ndims(imageRef) == 3 
        Y1 = 0.299 * double(imageRef(:,:,1)) + 0.587 * double(imageRef(:,:,2)) + 0.114 * double(imageRef(:,:,3));
        Y2 = 0.299 * double(imageDis(:,:,1)) + 0.587 * double(imageDis(:,:,2)) + 0.114 * double(imageDis(:,:,3));
        I1 = 0.596 * double(imageRef(:,:,1)) - 0.274 * double(imageRef(:,:,2)) - 0.322 * double(imageRef(:,:,3));
        I2 = 0.596 * double(imageDis(:,:,1)) - 0.274 * double(imageDis(:,:,2)) - 0.322 * double(imageDis(:,:,3));
        Q1 = 0.211 * double(imageRef(:,:,1)) - 0.523 * double(imageRef(:,:,2)) + 0.312 * double(imageRef(:,:,3));
        Q2 = 0.211 * double(imageDis(:,:,1)) - 0.523 * double(imageDis(:,:,2)) + 0.312 * double(imageDis(:,:,3));
    else
        Y1 = imageRef; Y2 = imageDis;
    end
    Y1 = double(Y1); Y2 = double(Y2);

    PC1 = phasecong2(Y1);
    PC2 = phasecong2(Y2);

    dx = [3 0 -3; 10 0 -10;  3  0 -3]/16;
    dy = [3 10 3; 0  0   0; -3 -10 -3]/16;
    IxY1 = conv2(Y1, dx, 'same'); IyY1 = conv2(Y1, dy, 'same');    
    gradientMap1 = sqrt(IxY1.^2 + IyY1.^2);
    IxY2 = conv2(Y2, dx, 'same'); IyY2 = conv2(Y2, dy, 'same');    
    gradientMap2 = sqrt(IxY2.^2 + IyY2.^2);

    T1 = 0.85; T2 = 160;
    PCSimMatrix = (2 * PC1 .* PC2 + T1) ./ (PC1.^2 + PC2.^2 + T1);
    gradientSimMatrix = (2*gradientMap1.*gradientMap2 + T2) ./(gradientMap1.^2 + gradientMap2.^2 + T2);
    PCm = max(PC1, PC2);
    SimMatrix = gradientSimMatrix .* PCSimMatrix .* PCm;
    FSIM = sum(sum(SimMatrix)) / sum(sum(PCm));

    T3 = 200; T4 = 200;
    ISimMatrix = (2 * I1 .* I2 + T3) ./ (I1.^2 + I2.^2 + T3);
    QSimMatrix = (2 * Q1 .* Q2 + T4) ./ (Q1.^2 + Q2.^2 + T4);
    lambda = 0.03;
    SimMatrixC = gradientSimMatrix .* PCSimMatrix .* real((ISimMatrix .* QSimMatrix) .^ lambda) .* PCm;
    FSIMc = sum(sum(SimMatrixC)) / sum(sum(PCm));
end % 主函数结束

function [ResultPC] = phasecong2(im)
    % 子函数1开始
    nscale = 4; norient = 4; minWaveLength = 6; mult = 2; sigmaOnf = 0.55; 
    dThetaOnSigma = 1.2; k = 2.0; epsilon = .0001;               
    thetaSigma = pi/norient/dThetaOnSigma;  
    [rows,cols] = size(im); imagefft = fft2(im); zero = zeros(rows,cols);
    EO = cell(nscale, norient); ifftFilterArray = cell(1,nscale); 
    if mod(cols,2), xrange = [-(cols-1)/2:(cols-1)/2]/(cols-1); else, xrange = [-cols/2:(cols/2-1)]/cols; end
    if mod(rows,2), yrange = [-(rows-1)/2:(rows-1)/2]/(rows-1); else, yrange = [-rows/2:(rows/2-1)]/rows; end
    [x,y] = meshgrid(xrange, yrange); radius = sqrt(x.^2 + y.^2); theta = atan2(-y,x);              
    radius = ifftshift(radius); theta = ifftshift(theta); radius(1,1) = 1;                  
    sintheta = sin(theta); costheta = cos(theta); 
    lp = lowpassfilter([rows,cols],.45,15); logGabor = cell(1,nscale);
    for s = 1:nscale
        wavelength = minWaveLength*mult^(s-1); fo = 1.0/wavelength;                  
        logGabor{s} = exp((-(log(radius/fo)).^2) / (2 * log(sigmaOnf)^2));  
        logGabor{s} = logGabor{s}.*lp; logGabor{s}(1,1) = 0;                 
    end
    spread = cell(1,norient);
    for o = 1:norient
      angl = (o-1)*pi/norient; ds = sintheta * cos(angl) - costheta * sin(angl);    
      dc = costheta * cos(angl) + sintheta * sin(angl); dtheta = abs(atan2(ds,dc));                          
      spread{o} = exp((-dtheta.^2) / (2 * thetaSigma^2));  
    end
    EnergyAll = zero; AnAll = zero;
    for o = 1:norient                    
      sumE_ThisOrient = zero; sumO_ThisOrient = zero; sumAn_ThisOrient = zero; Energy = zero;      
      for s = 1:nscale,                  
        filter = logGabor{s} .* spread{o}; ifftFilt = real(ifft2(filter))*sqrt(rows*cols);  
        ifftFilterArray{s} = ifftFilt; EO{s,o} = ifft2(imagefft .* filter);      
        An = abs(EO{s,o}); sumAn_ThisOrient = sumAn_ThisOrient + An;  
        sumE_ThisOrient = sumE_ThisOrient + real(EO{s,o}); sumO_ThisOrient = sumO_ThisOrient + imag(EO{s,o}); 
        if s==1, EM_n = sum(sum(filter.^2)); maxAn = An; else, maxAn = max(maxAn, An); end
      end                                       
      XEnergy = sqrt(sumE_ThisOrient.^2 + sumO_ThisOrient.^2) + epsilon;   
      MeanE = sumE_ThisOrient ./ XEnergy; MeanO = sumO_ThisOrient ./ XEnergy; 
      for s = 1:nscale,       
          E = real(EO{s,o}); O = imag(EO{s,o});    
          Energy = Energy + E.*MeanE + O.*MeanO - abs(E.*MeanO - O.*MeanE);
      end
      medianE2n = median(reshape(abs(EO{1,o}).^2,1,rows*cols)); meanE2n = -medianE2n/log(0.5); 
      noisePower = meanE2n/EM_n;                       
      EstSumAn2 = zero; for s = 1:nscale, EstSumAn2 = EstSumAn2 + ifftFilterArray{s}.^2; end
      EstSumAiAj = zero;
      for si = 1:(nscale-1)
        for sj = (si+1):nscale, EstSumAiAj = EstSumAiAj + ifftFilterArray{si}.*ifftFilterArray{sj}; end
      end
      sumEstSumAn2 = sum(sum(EstSumAn2)); sumEstSumAiAj = sum(sum(EstSumAiAj));
      EstNoiseEnergy2 = 2*noisePower*sumEstSumAn2 + 4*noisePower*sumEstSumAiAj;
      tau = sqrt(EstNoiseEnergy2/2); EstNoiseEnergy = tau*sqrt(pi/2); EstNoiseEnergySigma = sqrt( (2-pi/2)*tau^2 );
      T = (EstNoiseEnergy + k*EstNoiseEnergySigma)/1.7;        
      Energy = max(Energy - T, zero); EnergyAll = EnergyAll + Energy; AnAll = AnAll + sumAn_ThisOrient;
    end  
    ResultPC = EnergyAll ./ AnAll;
end % 子函数1结束

function f = lowpassfilter(sze, cutoff, n)
    % 子函数2开始
    if length(sze) == 1, rows = sze; cols = sze; else, rows = sze(1); cols = sze(2); end
    if mod(cols,2), xrange = [-(cols-1)/2:(cols-1)/2]/(cols-1); else, xrange = [-cols/2:(cols/2-1)]/cols; end
    if mod(rows,2), yrange = [-(rows-1)/2:(rows-1)/2]/(rows-1); else, yrange = [-rows/2:(rows/2-1)]/rows; end
    [x,y] = meshgrid(xrange, yrange); radius = sqrt(x.^2 + y.^2);        
    f = ifftshift( 1 ./ (1.0 + (radius ./ cutoff).^(2*n)) );   
end % 子函数2结束
