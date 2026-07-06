import torch
import torch.nn as nn
import torchaudio
import torch.nn.functional as F
import soundfile as sf


class AudioCNNCoach(nn.Module):
    def __init__(self, num_clases=3):
        super(AudioCNNCoach, self).__init__()

        self.bloque1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.bloque2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.flatten = nn.Flatten()

        with torch.no_grad():
            tensor_prueba = torch.zeros(1, 1, 201, 201)
            x_prueba = self.bloque1(tensor_prueba)
            x_prueba = self.bloque2(x_prueba)
            x_prueba = self.flatten(x_prueba)
            tamanio_aplanado = x_prueba.shape[1]

        self.clasificador = nn.Sequential(
            nn.Linear(tamanio_aplanado, 64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, num_clases)
        )

    def forward(self, x):
        x = self.bloque1(x)
        x = self.bloque2(x)
        x = self.flatten(x)
        x = self.clasificador(x)
        return x


def preparar_audio_nuevo(ruta_audio, duracion_segundos=2, sample_rate=16000):
    try:
        waveform, sr = torchaudio.load(ruta_audio, backend="soundfile")
        waveform = waveform.float()
    except Exception:
        data, sr = sf.read(ruta_audio, dtype="float32")
        if data.ndim == 1:
            data = data.reshape(1, -1)
        else:
            data = data.T
        waveform = torch.from_numpy(data).float()

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resampler(waveform)

    muestras_objetivo = sample_rate * duracion_segundos

    if waveform.shape[1] > muestras_objetivo:
        waveform = waveform[:, :muestras_objetivo]
    elif waveform.shape[1] < muestras_objetivo:
        pad_amount = muestras_objetivo - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad_amount))

    transformador = torchaudio.transforms.Spectrogram(n_fft=400, hop_length=160)
    espectrograma = transformador(waveform)
    espectrograma = torch.log2(espectrograma + 1e-6)


    espectrograma = espectrograma.unsqueeze(0)
    return espectrograma


def analizar_canto(ruta_audio, ruta_modelo="modelos/coach_vocal_cnn.pth"):
    print(f"\nProcesando audio: {ruta_audio} ...")


    modelo = AudioCNNCoach(num_clases=3)
    try:
        modelo.load_state_dict(torch.load(ruta_modelo, weights_only=True))
        modelo.eval()
    except FileNotFoundError:
        print("Error: No se encontro el archivo del modelo entrenado en la ruta especificada.")
        return


    espectrograma = preparar_audio_nuevo(ruta_audio)


    clases_nombres = {0: "Breathy (Aireada)", 1: "Belt (Potencia)", 2: "Straight (Recta/Lirica)"}


    with torch.no_grad():
        salida = modelo(espectrograma)
        probabilidades = F.softmax(salida, dim=1)

        prob_max, indice_predicho = torch.max(probabilidades, dim=1)
        id_clase = indice_predicho.item()
        confianza = prob_max.item() * 100

    print("-" * 50)
    print(f"ANALISIS DE LA IA:")
    print(f"Tecnica detectada: {clases_nombres[id_clase]}")
    print(f"Confianza del modelo: {confianza:.2f}%")
    print("-" * 50)


    print("COACH VOCAL IA:")
    if id_clase == 0:
        print("He detectado mucha fuga de aire en tu emision (Breathy).")
        print("Diagnostico: Falta de cierre cordal adecuado.")
        print("Ejercicio: Canta staccatos con la silaba 'GUG' o 'BUP' en arpegios para ayudar a que tus cuerdas vocales se junten de manera firme sin tension.")

    elif id_clase == 1:
        print("He detectado una emision potente y con mucho cuerpo (Belt).")
        print("Diagnostico: Excelente resonancia, pero requiere cuidado para no lastimar la laringe.")
        print("Consejo: Asegurate de que el soporte venga del diafragma y no del empuje desde la garganta. Relaja tu voz haciendo ejercicios de sirenas (Lip Trills) de agudo a grave al terminar de cantar.")

    elif id_clase == 2:
        print("He detectado un tono muy limpio, controlado y sin oscilacion (Straight).")
        print("Diagnostico: Tienes un buen balance tonal, tipico del canto lirico o pop suave.")
        print("Consejo: Si notas que la nota se vuelve tensa al sostenerla, intenta liberar un poco la laringe para permitir que aparezca el vibrato natural. Trata de pensar en el sonido girando en lugar de ir en linea recta.")

    print("-" * 50)




if __name__ == "__main__":



    archivo_prueba = "/home/nicole05/Documentos/IA/Passaggio/datos/belt/f1_arpeggios_belt_c_a.wav"
    analizar_canto(archivo_prueba)
