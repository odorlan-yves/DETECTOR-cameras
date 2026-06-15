import cv2
from ultralytics import YOLO
import easyocr
import numpy as np
import re
import os
from datetime import datetime

# ==============================================================================
# CONFIGURAÇÕES
# ==============================================================================
MODEL_PATH = "best.pt"         # Caminho do seu modelo treinado
CONF_THRESHOLD = 0.5           # Limiar de confiança mínimo para a detecção
ARQUIVO_HISTORICO = "historico_placas.txt"  # Arquivo de texto onde o histórico será salvo
# ==============================================================================


# ═══════════════════════════════════════════════════════════════════════════════
# FUNÇÕES DE TRATAMENTO DE IMAGEM, VALIDAÇÃO E ARQUIVO
# ═══════════════════════════════════════════════════════════════════════════════
def salvar_no_historico(texto_placa: str):
    """Grava a placa e o horário atual no arquivo de texto simples (.txt)."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # O modo 'a' (append) abre o arquivo e adiciona o texto no final dele sem apagar o que já existe
    with open(ARQUIVO_HISTORICO, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] Placa: {texto_placa}\n")


def obter_ultimas_linhas(limite=5):
    """Lê o arquivo de texto e retorna as últimas linhas para exibir na tela."""
    if not os.path.exists(ARQUIVO_HISTORICO):
        return []
    
    with open(ARQUIVO_HISTORICO, "r", encoding="utf-8") as f:
        linhas = f.readlines()
        
    # Retorna as últimas linhas invertidas (a mais recente primeiro)
    return [l.strip() for l in linhas[-limite:]][::-1]


def limpar_e_validar_texto(texto: str):
    """Remove sujeiras do texto e valida se possui o tamanho mínimo de uma placa."""
    texto_limpo = re.sub(r"[^A-Z0-9]", "", texto.upper())
    
    # Validação padrão Mercosul/Antiga (7 caracteres)
    if len(texto_limpo) == 7:
        return texto_limpo, True
    
    # Aceita leituras parciais se tiver pelo menos 5 caracteres
    if len(texto_limpo) >= 5:
        return texto_limpo, False
        
    return None, False


def preprocessar_recorte(recorte: np.ndarray) -> np.ndarray:
    """Melhora o contraste do recorte da placa para facilitar a leitura do OCR."""
    if recorte.size == 0:
        return recorte
    h_alvo = 100
    proporcao = h_alvo / max(recorte.shape[0], 1)
    recorte_redim = cv2.resize(recorte, (int(recorte.shape[1] * proporcao), h_alvo))
    
    cinza = cv2.cvtColor(recorte_redim, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(6, 6))
    return clahe.apply(cinza)


# ═══════════════════════════════════════════════════════════════════════════════
# EXECUÇÃO PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print(f"Carregando modelo YOLO: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    print("Inicializando leitor OCR (EasyOCR)...")
    # gpu=False força o uso da CPU. Mude para True se tiver GPU Nvidia instalada
    leitor_ocr = easyocr.Reader(["pt"], gpu=False, verbose=False)

    ultima_placa_salva = None
    tempo_ultimo_ocr = 0

    print("Abrindo webcam...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Erro: não foi possível abrir a webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\nSistema Pronto! Pressione 'q' ou ESC para sair.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erro ao ler frame da webcam.")
            break

        if frame is None or frame.size == 0:
            continue

        frame_render = frame.copy()

        # 1. Inferência com o seu modelo treinado
        results = model(frame, conf=CONF_THRESHOLD, verbose=False)
        result = results[0]

        # Mantém apenas a detecção de maior confiança
        if result.boxes is not None and len(result.boxes) > 0:
            confidences = result.boxes.conf.cpu().numpy()
            best_idx = confidences.argmax()
            
            box = result.boxes[best_idx]
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            confianca_yolo = float(box.conf[0].cpu().numpy())

            # Desenha o retângulo na placa
            cv2.rectangle(frame_render, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # 2. Processo de OCR (Leitura do texto a cada 0.5 segundos)
            agora = cv2.getTickCount() / cv2.getTickFrequency()
            if (agora - tempo_ultimo_ocr) > 0.5:
                tempo_ultimo_ocr = agora
                
                h, w = frame.shape[:2]
                recorte = frame[max(0, y1-4):min(h, y2+4), max(0, x1-4):min(w, x2+4)]
                
                if recorte.size > 0:
                    imagem_preparada = preprocessar_recorte(recorte)
                    leituras = leitor_ocr.readtext(
                        imagem_preparada, 
                        detail=0, 
                        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
                    )
                    
                    if leituras:
                        texto_bruto = "".join(leituras)
                        texto_validado, eh_perfeita = limpar_e_validar_texto(texto_bruto)
                        
                        if texto_validado:
                            # Filtro para não registrar repetidamente a mesma placa enquanto ela estiver parada na câmera
                            if texto_validado != ultima_placa_salva:
                                salvar_no_historico(texto_validado)
                                ultima_placa_salva = texto_validado
                                print(f"[SALVO NO TXT] Placa: {texto_validado} (Confiança: {confianca_yolo:.2f})")

            # Exibe a string da placa no topo do retângulo
            if ultima_placa_salva:
                cv2.putText(
                    frame_render, f"PLACA: {ultima_placa_salva}", 
                    (x1, max(y1 - 10, 25)), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, (0, 255, 0), 2, cv2.LINE_AA
                )

        # ═══════════════════════════════════════════════════════════════════════
        # PAINEL VISUAL DO HISTÓRICO (.TXT)
        # ═══════════════════════════════════════════════════════════════════════
        # Desenha um fundo escuro no canto superior esquerdo
        cv2.rectangle(frame_render, (0, 0), (350, 240), (15, 15, 15), -1)
        cv2.putText(frame_render, "ULTIMAS PLACAS (HISTORICO.TXT):", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Lê as últimas linhas salvas direto do arquivo TXT e joga na tela
        ultimos_txt = obter_ultimas_linhas(5)
        pos_y = 60
        for linha_texto in ultimos_txt:
            # Encurta o tamanho para caber no menu se a linha for muito grande
            texto_exibicao = linha_texto.replace("Placa: ", "")
            cv2.putText(
                frame_render, texto_exibicao, 
                (15, pos_y), cv2.FONT_HERSHEY_SIMPLEX, 
                0.50, (200, 255, 200), 1, cv2.LINE_AA
            )
            pos_y += 35

        cv2.imshow("YOLO + EasyOCR (Salva em TXT)", frame_render)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()