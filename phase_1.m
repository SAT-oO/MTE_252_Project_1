

function y = process(file)

    [y, Fs] = audioread(file);
    [~, channels] = size(y);

    if Fs < 16000
        error('Error: Sample Rate too Slow!');
    end

    if channels == 2
        % average to retain magnitude
        y = (y(:, 1) + y(:, 2)) / 2;
    end

    sound(y, Fs);

    % generate a mono-channel output
    audiowrite('output.wav', y, Fs);

    plot(y);
    xlabel('Sample Number'); 
    ylabel('Amplitude');
    title('Sample Waveform');

    % resample to 16 kHz
    if Fs ~= 16000
        y = resample(y, 16000, Fs);
        Fs = 16000;
    end

    % generate 1 kHz cos wave

    N = length(y);
    t = linspace(0:N-1) / Fs;
    c = cos(2000*pi*t);

    % 2 ms => 2 ms * 16kHz = 32 samples

    plot(t(1:32) * 1000, c(1:32), '-o');
    xlabel('Time (ms)');
    ylabel('Amplitude');
    title('1 kHz cosine wave');
    
end

