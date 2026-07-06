import os
import torch
import torch.nn as nn
import torchaudio
import torch.nn.functional as F
import soundfile as sf
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename


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
            tamanio_aplanado = self.flatten(x_prueba).shape[1]

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
    """Carga y preprocesa audio con fallback a soundfile para mayor compatibilidad."""
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
        waveform = F.pad(waveform, (0, pad_amount))

    transformador = torchaudio.transforms.Spectrogram(n_fft=400, hop_length=160)
    espectrograma = transformador(waveform)
    espectrograma = torch.log2(espectrograma + 1e-6)
    return espectrograma.unsqueeze(0)


modelo_ia = AudioCNNCoach(num_clases=3)
modelo_ia.load_state_dict(torch.load("modelos/coach_vocal_cnn.pth", weights_only=True))
modelo_ia.eval()

clases_info = {
    0: {
        "nombre": "Breathy (Fuga de aire)",
        "diagnostico": "Falta de cierre cordal adecuado.",
        "consejo": "Canta staccatos con la silaba 'GUG' o 'BUP' en arpegios para ayudar a que tus cuerdas vocales se junten firmemente sin tension."
    },
    1: {
        "nombre": "Belt (Potencia)",
        "diagnostico": "Excelente resonancia, pero requiere cuidado para no lastimar la laringe.",
        "consejo": "Asegurate de que el soporte venga del diafragma y no del empuje desde la garganta. Haz ejercicios de sirenas (Lip Trills) de agudo a grave al terminar."
    },
    2: {
        "nombre": "Straight (Recta/Lirica)",
        "diagnostico": "Buen balance tonal, tipico del canto lirico o pop suave.",
        "consejo": "Si notas tension al sostener la nota, libera la laringe permitiendo el vibrato natural. Piensa en el sonido girando, no en linea recta."
    }
}


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', resultado=None)

@app.route('/analizar', methods=['POST'])
def analizar():
    if 'audio_file' not in request.files:
        return render_template('index.html', error="No se envio ningun archivo.")

    file = request.files['audio_file']
    if file.filename == '':
        return render_template('index.html', error="No se selecciono ningun archivo.")

    if file:

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)


        try:
            espectrograma = preparar_audio_nuevo(filepath)
            with torch.no_grad():
                salida = modelo_ia(espectrograma)
                probabilidades = F.softmax(salida, dim=1)
                prob_max, indice_predicho = torch.max(probabilidades, dim=1)

                id_clase = indice_predicho.item()
                confianza = prob_max.item() * 100

            info = clases_info[id_clase]
            resultado = {
                "tecnica": info["nombre"],
                "confianza": f"{confianza:.2f}%",
                "diagnostico": info["diagnostico"],
                "consejo": info["consejo"]
            }
        except Exception as e:
            return render_template('index.html', error=f"Error al procesar el audio: {str(e)}")
        finally:

            if os.path.exists(filepath):
                os.remove(filepath)

        return render_template('index.html', resultado=resultado)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
