process('child_approx.wav');

% figures are attached to document with part 1

function y = process(file)

    [~, name, ~] = fileparts(file);
    file_name = name

    [y, Fs] = audioread(file);
    sampling_rate = Fs
    [~, channels] = size(y);

    if Fs < 16000
        error('Error: Sample Rate too Slow!');
    end

    if channels == 2
        % average to retain magnitude
        y = (y(:, 1) + y(:, 2)) / 2;
    end

    % plot waveform
    figure;
    plot(y);
    xlabel('Sample Number'); 
    ylabel('Amplitude');
    title('Sample Waveform');

    % resample to 16 kHz
    if Fs ~= 16000
        y = resample(y, 16000, Fs);
        Fs = 16000;
    end

    % processed sound
    sound(y, Fs);
    % this prevents next sound from being played instantly
    pause(length(y)/Fs + 0.5);
    
    % save output
    audiowrite([name '_out.wav'], y, Fs);

    % generate 1 kHz cos wave

    N = length(y);
    t = linspace(0, (N-1) / Fs, N)';
    c = cos(2000*pi*t);

    % play cosine wave
    sound(c, Fs);
    pause(N/Fs + 0.5);

    % 2 ms => 2 ms * 16kHz = 32 + 1 samples

    figure;
    plot(t(1:33) * 1000, c(1:33), '-o');
    xlabel('Time (ms)');
    ylabel('Amplitude');
    title('1 kHz cosine wave');
    
end

