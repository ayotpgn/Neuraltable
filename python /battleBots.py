import random
import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# --- Importação e Configuração do Flask (Se Disponível) ---
app = None
FLASK_AVAILABLE = False
try:
    from flask import Flask, render_template, jsonify, request
    from threading import Thread
    FLASK_AVAILABLE = True
    app = Flask(__name__, template_folder='.')
except ImportError:
    print("Flask não está instalado. A interface web opcional não estará disponível.")
    print("Para instalar o Flask, execute: pip install Flask")

# Este bloco só é executado se 'app' foi criado com sucesso (Flask disponível)
if app:
    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/game_state')
    def get_game_state():
        return jsonify(game_state_web)

    @app.route('/player_move', methods=['POST'])
    def player_move():
        data = request.json
        print(f"Recebido movimento do jogador: {data}")
        return jsonify({"status": "Move received", "data": data})

def run_flask_app():
    if FLASK_AVAILABLE and app:
        app.run(host='0.0.0.0', port=5000)
    else:
        print("Não é possível iniciar a interface web: Flask não está disponível.")
# --- Fim da Configuração do Flask ---

# --- Global Message Log ---
MESSAGE_LOG = deque(maxlen=5)
# --- End Global Message Log ---

# --- Game Configuration ---
BOARD_SIZE = 7
UNITS_PER_BOT = 3
MAX_TURNS = 1200
EPISODES = 1000
TILE_SIZE = 60
MODEL_DIR = "models"
PERFORMANCE_DIR = "performance"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PERFORMANCE_DIR, exist_ok=True)

# --- Variáveis Globais para Imagens (Carregadas apenas uma vez) ---
CORINTHIANS_UNIT_IMAGE = None
PALMEIRAS_UNIT_IMAGE = None
PLAYER_UNIT_IMAGE = None
# --- Fim Variáveis Globais ---

# --- PyTorch DQN ---
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.model(x)

# --- Replay Buffer ---
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

# --- Game Unit ---
class Unit:
    def __init__(self, x, y, hp=3):
        self.x = x
        self.y = y
        self.hp = hp

    def is_alive(self):
        return self.hp > 0

# --- DQN Agent ---
class Bot:
    def __init__(self, name, color, input_dim, action_dim):
        self.name = name
        self.color = color
        self.model = DQN(input_dim, action_dim)
        self.target_model = DQN(input_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.buffer = ReplayBuffer()
        self.units = []
        self.epsilon = 1.0
        self.action_dim = action_dim
        self.update_target()

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def get_alive_units(self):
        return [u for u in self.units if u.is_alive()]

    def encode_state(self, enemy_units):
        state = []
        for u in self.get_alive_units():
            state += [u.x / BOARD_SIZE, u.y / BOARD_SIZE, u.hp / 3]
        for _ in range(UNITS_PER_BOT - len(self.get_alive_units())):
            state += [0, 0, 0]
        for u in enemy_units:
            state += [u.x / BOARD_SIZE, u.y / BOARD_SIZE, u.hp / 3]
        for _ in range(UNITS_PER_BOT - len(enemy_units)):
            state += [0, 0, 0]
        return np.array(state, dtype=np.float32)

    def choose_action(self, state_tensor):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            q_values = self.model(torch.tensor(state_tensor).unsqueeze(0))
            return q_values.argmax().item()

    def train_step(self, batch_size=32, gamma=0.99):
        if len(self.buffer) < batch_size:
            return 0.0
        state, action, reward, next_state, done = self.buffer.sample(batch_size)
        state = torch.tensor(state)
        action = torch.tensor(action)
        reward = torch.tensor(reward)
        next_state = torch.tensor(next_state)
        done = torch.tensor(done, dtype=torch.float32)

        q_vals = self.model(state).gather(1, action.unsqueeze(1)).squeeze()
        next_q_vals = self.target_model(next_state).max(1)[0]
        expected = reward + gamma * next_q_vals * (1 - done)

        loss = nn.MSELoss()(q_vals, expected.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Modelo de {self.name} salvo em {path}")

    def load_model(self, path):
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path))
            self.update_target()
            print(f"Modelo de {self.name} carregado de {path}")
            return True
        print(f"Nenhum modelo encontrado em {path} para {self.name}")
        return False

    def send_message(self, msg):
        full_msg = f"[{self.name}] {msg}"
        MESSAGE_LOG.append(full_msg)
        print(full_msg)

# --- Game Functions (MOVED TO HERE) ---

# Função para posicionamento estratégico das unidades
def place_units_strategically(bot1, bot2, board_size):
    bot1.units = []
    bot2.units = []
    all_positions = set()

    while len(bot1.units) < UNITS_PER_BOT:
        x = random.randint(0, board_size // 3 - 1)
        y = random.randint(0, board_size // 3 - 1)
        if (x, y) not in all_positions:
            bot1.units.append(Unit(x, y))
            all_positions.add((x, y))

    while len(bot2.units) < UNITS_PER_BOT:
        x = random.randint(board_size - board_size // 3, board_size - 1)
        y = random.randint(board_size - board_size // 3, board_size - 1)
        if (x, y) not in all_positions:
            bot2.units.append(Unit(x, y))
            all_positions.add((x, y))

# Função para carregar e pré-processar as imagens
def load_unit_images():
    global CORINTHIANS_UNIT_IMAGE, PALMEIRAS_UNIT_IMAGE, PLAYER_UNIT_IMAGE
    try:
        # Tente carregar as imagens do disco
        cor_img = pygame.image.load(os.path.join(os.path.dirname(__file__), "corinthians_logo.png")).convert_alpha()
        pal_img = pygame.image.load(os.path.join(os.path.dirname(__file__), "palmeiras_logo.png")).convert_alpha()
        player_img = pygame.image.load(os.path.join(os.path.dirname(__file__), "player_logo.png")).convert_alpha() # Certifique-se de ter este arquivo ou crie um

        CORINTHIANS_UNIT_IMAGE = pygame.transform.scale(cor_img, (TILE_SIZE, TILE_SIZE))
        PALMEIRAS_UNIT_IMAGE = pygame.transform.scale(pal_img, (TILE_SIZE, TILE_SIZE))
        PLAYER_UNIT_IMAGE = pygame.transform.scale(player_img, (TILE_SIZE, TILE_SIZE))

        print("Imagens de unidades carregadas com sucesso!")
    except pygame.error as e:
        print(f"Erro ao carregar imagens: {e}. Desenhando círculos em vez de imagens.")
        print("Certifique-se de que 'corinthians_logo.png', 'palmeiras_logo.png' e 'player_logo.png' estão na mesma pasta do script.")
        # Define as imagens como None para que draw_board desenhe círculos
        CORINTHIANS_UNIT_IMAGE = None
        PALMEIRAS_UNIT_IMAGE = None
        PLAYER_UNIT_IMAGE = None
    except FileNotFoundError:
        print("Arquivos de imagem não encontrados. Desenhando círculos em vez de imagens.")
        print("Certifique-se de que 'corinthians_logo.png', 'palmeiras_logo.png' e 'player_logo.png' estão na mesma pasta do script.")
        CORINTHIANS_UNIT_IMAGE = None
        PALMEIRAS_UNIT_IMAGE = None
        PLAYER_UNIT_IMAGE = None


def draw_board(screen, bot1, bot2, current_turn_info=""):
    screen_width = BOARD_SIZE * TILE_SIZE
    screen_height = BOARD_SIZE * TILE_SIZE
    if screen.get_width() != screen_width or screen.get_height() != screen_height:
        pygame.display.set_mode((screen_width, screen_height))

    screen.fill((255, 255, 255))
    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            pygame.draw.rect(screen, (0, 0, 0), (x*TILE_SIZE, y*TILE_SIZE, TILE_SIZE, TILE_SIZE), 1)

    font = pygame.font.Font(None, 24)

    # Desenhar unidades com imagens ou círculos de fallback
    for u in bot1.get_alive_units():
        if bot1.name == "Corinthians" and CORINTHIANS_UNIT_IMAGE:
            screen.blit(CORINTHIANS_UNIT_IMAGE, (u.x*TILE_SIZE, u.y*TILE_SIZE))
        elif bot1.name == "Jogador" and PLAYER_UNIT_IMAGE:
            screen.blit(PLAYER_UNIT_IMAGE, (u.x*TILE_SIZE, u.y*TILE_SIZE))
        else:
            pygame.draw.circle(screen, bot1.color, (u.x*TILE_SIZE+TILE_SIZE//2, u.y*TILE_SIZE+TILE_SIZE//2), TILE_SIZE//3)
        
        # Desenhar HP em cima da unidade
        hp_text = font.render(str(u.hp), True, (255, 255, 255))
        hp_text_rect = hp_text.get_rect(center=(u.x*TILE_SIZE+TILE_SIZE//2, u.y*TILE_SIZE+TILE_SIZE//2))
        screen.blit(hp_text, hp_text_rect)


    for u in bot2.get_alive_units():
        if bot2.name == "Palmeiras" and PALMEIRAS_UNIT_IMAGE:
            screen.blit(PALMEIRAS_UNIT_IMAGE, (u.x*TILE_SIZE, u.y*TILE_SIZE))
        elif bot2.name == "Jogador" and PLAYER_UNIT_IMAGE:
            screen.blit(PLAYER_UNIT_IMAGE, (u.x*TILE_SIZE, u.y*TILE_SIZE))
        else:
            pygame.draw.circle(screen, bot2.color, (u.x*TILE_SIZE+TILE_SIZE//2, u.y*TILE_SIZE+TILE_SIZE//2), TILE_SIZE//3)
        
        # Desenhar HP em cima da unidade
        hp_text = font.render(str(u.hp), True, (255, 255, 255))
        hp_text_rect = hp_text.get_rect(center=(u.x*TILE_SIZE+TILE_SIZE//2, u.y*TILE_SIZE+TILE_SIZE//2))
        screen.blit(hp_text, hp_text_rect)

    info_text = font.render(current_turn_info, True, (0, 0, 0))
    screen.blit(info_text, (5, 5))

    msg_y_offset = 30
    for i, msg in enumerate(MESSAGE_LOG):
        msg_text = font.render(msg, True, (0, 0, 0))
        screen.blit(msg_text, (5, screen_height - msg_y_offset - (len(MESSAGE_LOG) - 1 - i) * 20))

    pygame.display.flip()

def make_move(player_bot, enemy_bot, is_player_turn=False, player_action=None):
    alive_units = player_bot.get_alive_units()
    if not alive_units:
        return False, 0, "Nenhuma unidade viva para mover."

    state = player_bot.encode_state(enemy_bot.get_alive_units())

    if is_player_turn:
        if player_action is None:
            return False, 0, "Ação do jogador necessária."
        action = player_action
        unit_index = action // 5
        if unit_index >= len(alive_units):
            player_bot.send_message(f"Tentou selecionar unidade {unit_index} que não existe ou está morta.")
            return False, 0, "Unidade selecionada inválida ou morta."
        selected_unit = alive_units[unit_index]
        action_type = action % 5
    else:
        action = player_bot.choose_action(state)
        unit_index = action // 5
        if unit_index >= len(alive_units):
            player_bot.send_message(f"IA tentou selecionar unidade {unit_index} que não existe ou está morta.")
            return False, -0.5, "Ação de IA inválida (unidade morta ou índice fora do alcance)."

        selected_unit = alive_units[unit_index]
        action_type = action % 5


    original_unit_pos = (selected_unit.x, selected_unit.y)
    original_enemy_hp = {id(u): u.hp for u in enemy_bot.get_alive_units()}

    reward_for_turn = 0
    message = ""

    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    if action_type < 4:
        dx, dy = directions[action_type]
        new_x = max(0, min(BOARD_SIZE - 1, selected_unit.x + dx))
        new_y = max(0, min(BOARD_SIZE - 1, selected_unit.y + dy))

        collision = False
        for u in alive_units:
            if u is not selected_unit and u.x == new_x and u.y == new_y:
                collision = True
                break
        
        if not collision:
            selected_unit.x = new_x
            selected_unit.y = new_y
            message = f"Movi unidade {unit_index} para ({selected_unit.x}, {selected_unit.y})"
        else:
            message = f"Tentei mover unidade {unit_index} para ({new_x}, {new_y}) mas colidi."
            reward_for_turn -= 0.1
    else:
        attacked = False
        for e in enemy_bot.get_alive_units():
            if abs(selected_unit.x - e.x) + abs(selected_unit.y - e.y) == 1:
                e.hp -= 1
                attacked = True
                message = f"Ataquei unidade inimiga em ({e.x}, {e.y}). HP restante: {e.hp}"
                break
        if not attacked:
            message = f"Tentei atacar com unidade {unit_index} mas não encontrei alvo adjacente."
            reward_for_turn -= 0.5

    next_state = player_bot.encode_state(enemy_bot.get_alive_units())

    if not enemy_bot.get_alive_units():
        reward_for_turn += 10
        done = True
    elif not player_bot.get_alive_units():
        reward_for_turn -= 10
        done = True
    else:
        done = False
        current_enemy_hp_sum = sum(u.hp for u in enemy_bot.get_alive_units())
        previous_enemy_hp_sum = sum(original_enemy_hp.values())
        reward_for_turn += (previous_enemy_hp_sum - current_enemy_hp_sum) * 1.0
        reward_for_turn -= 0.01
    
    player_bot.send_message(message)
    pygame.time.wait(500)

    player_bot.buffer.push(state, action, reward_for_turn, next_state, done)
    return done, reward_for_turn, message

# --- Matplotlib Performance Graphics ---
def plot_performance(rewards_history, losses_history, filename="training_performance.png"):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(rewards_history)
    plt.title('Recompensa por Episódio')
    plt.xlabel('Episódio')
    plt.ylabel('Recompensa Total')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(losses_history)
    plt.title('Perda por Episódio')
    plt.xlabel('Episódio')
    plt.ylabel('Perda Média')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(PERFORMANCE_DIR, filename))
    plt.close()
    print(f"Gráfico de desempenho salvo em {os.path.join(PERFORMANCE_DIR, filename)}")

def plot_game_results(results_history, filename="game_results.png"):
    if not results_history:
        print("Nenhum resultado de jogo para plotar.")
        return

    corinthians_wins = results_history.count('Corinthians')
    palmeiras_wins = results_history.count('Palmeiras')
    draws = results_history.count('Draw')
    player_wins = results_history.count('Jogador')

    labels = ['Corinthians Wins', 'Palmeiras Wins', 'Empates']
    counts = [corinthians_wins, palmeiras_wins, draws]

    if player_wins > 0:
        labels.append('Jogador Wins')
        counts.append(player_wins)

    colors = ['red', 'green', 'gray', 'purple']

    plt.figure(figsize=(8, 6))
    plt.bar(labels, counts, color=colors[:len(labels)])
    plt.title('Resultados dos Jogos')
    plt.xlabel('Resultado')
    plt.ylabel('Número de Episódios')
    plt.grid(axis='y')

    for i, count in enumerate(counts):
        plt.text(i, count + 0.1, str(count), ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(PERFORMANCE_DIR, filename))
    plt.close()
    print(f"Gráfico de resultados do jogo salvo em {os.path.join(PERFORMANCE_DIR, filename)}")


# --- Pygame Interface Update ---
def get_pygame_input(bot1, bot2):
    pass

# --- Global game state for web interface ---
game_state_web = {
    'board': [],
    'bot1_units': [],
    'bot2_units': [],
    'current_turn_info': "",
    'game_over': False,
    'winner': None
}

def update_web_game_state(bot1, bot2, current_turn_info="", game_over=False, winner=None):
    global game_state_web
    board_data = [[0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]

    bot1_units_data = []
    for u in bot1.get_alive_units():
        board_data[u.y][u.x] = 1
        bot1_units_data.append({'x': u.x, 'y': u.y, 'hp': u.hp, 'team': bot1.name})
        
    bot2_units_data = []
    for u in bot2.get_alive_units():
        board_data[u.y][u.x] = 2
        bot2_units_data.append({'x': u.x, 'y': u.y, 'hp': u.hp, 'team': bot2.name})

    game_state_web = {
        'board': board_data,
        'bot1_units': bot1_units_data,
        'bot2_units': bot2_units_data,
        'current_turn_info': current_turn_info,
        'game_over': game_over,
        'winner': winner
    }


# --- Training Mode ---
def train_mode():
    pygame.init()
    # Carregar imagens uma vez no início do modo
    load_unit_images() # CHAMADA AQUI!
    screen_width = BOARD_SIZE * TILE_SIZE
    screen_height = BOARD_SIZE * TILE_SIZE
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("IA vs IA - Corinthians vs Palmeiras")
    
    input_dim = UNITS_PER_BOT * 3 * 2
    action_dim = UNITS_PER_BOT * 5

    corinthians_bot = Bot("Corinthians", (255, 0, 0), input_dim, action_dim)
    palmeiras_bot = Bot("Palmeiras", (0, 128, 0), input_dim, action_dim)

    corinthians_bot.load_model(os.path.join(MODEL_DIR, "bot_corinthians_model.pth"))
    palmeiras_bot.load_model(os.path.join(MODEL_DIR, "bot_palmeiras_model.pth"))

    rewards_history = []
    losses_history = []
    game_results_history = []

    for episode in range(EPISODES):
        place_units_strategically(corinthians_bot, palmeiras_bot, BOARD_SIZE)
        MESSAGE_LOG.clear()
        done = False
        episode_reward_bot1 = 0
        episode_reward_bot2 = 0
        episode_losses = []
        episode_winner = "Nenhum"

        for turn in range(MAX_TURNS):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    corinthians_bot.save_model(os.path.join(MODEL_DIR, "bot_corinthians_model.pth"))
                    palmeiras_bot.save_model(os.path.join(MODEL_DIR, "bot_palmeiras_model.pth"))
                    plot_performance(rewards_history, losses_history, "training_performance.png")
                    plot_game_results(game_results_history, "training_game_results.png")
                    pygame.quit()
                    return

            current_turn_info = f"Episódio: {episode+1}/{EPISODES} | Turno: {turn+1}/{MAX_TURNS}"
            draw_board(screen, corinthians_bot, palmeiras_bot, current_turn_info)
            update_web_game_state(corinthians_bot, palmeiras_bot, current_turn_info)

            if not done and corinthians_bot.get_alive_units():
                done_bot1, reward_bot1, _ = make_move(corinthians_bot, palmeiras_bot)
                episode_reward_bot1 += reward_bot1
                if done_bot1: done = True
            
            if not done and palmeiras_bot.get_alive_units():
                done_bot2, reward_bot2, _ = make_move(palmeiras_bot, corinthians_bot)
                episode_reward_bot2 += reward_bot2
                if done_bot2: done = True

            loss1 = corinthians_bot.train_step()
            loss2 = palmeiras_bot.train_step()
            if loss1 > 0: episode_losses.append(loss1)
            if loss2 > 0: episode_losses.append(loss2)
            
            if done:
                print(f"Episódio {episode+1} terminou no turno {turn+1}. Corinthians Alive: {len(corinthians_bot.get_alive_units())}, Palmeiras Alive: {len(palmeiras_bot.get_alive_units())}")
                if not palmeiras_bot.get_alive_units() and corinthians_bot.get_alive_units():
                    corinthians_bot.send_message("Eu venci!")
                    episode_winner = "Corinthians"
                    update_web_game_state(corinthians_bot, palmeiras_bot, "Corinthians Venceu!", True, "Corinthians")
                elif not corinthians_bot.get_alive_units() and palmeiras_bot.get_alive_units():
                    palmeiras_bot.send_message("Eu venci!")
                    episode_winner = "Palmeiras"
                    update_web_game_state(corinthians_bot, palmeiras_bot, "Palmeiras Venceu!", True, "Palmeiras")
                else:
                    corinthians_bot.send_message("Parece que foi um empate!")
                    palmeiras_bot.send_message("Empate!")
                    episode_winner = "Draw"
                    update_web_game_state(corinthians_bot, palmeiras_bot, "Empate!", True, "Draw")
                pygame.time.wait(1000)
                break 
        
        if not done:
             if len(corinthians_bot.get_alive_units()) > len(palmeiras_bot.get_alive_units()):
                episode_winner = "Corinthians"
                corinthians_bot.send_message("Eu venci por unidades restantes!")
                print(f"Episódio {episode+1} terminou por limite de turnos. Corinthians Venceu por unidades restantes!")
             elif len(palmeiras_bot.get_alive_units()) > len(corinthians_bot.get_alive_units()):
                episode_winner = "Palmeiras"
                palmeiras_bot.send_message("Eu venci por unidades restantes!")
                print(f"Episódio {episode+1} terminou por limite de turnos. Palmeiras Venceu por unidades restantes!")
             else:
                episode_winner = "Draw"
                corinthians_bot.send_message("Fim do tempo. Foi um empate!")
                print(f"Episódio {episode+1} terminou por limite de turnos. Empate por unidades restantes iguais!")

        game_results_history.append(episode_winner)

        rewards_history.append(episode_reward_bot1)
        if episode_losses:
            losses_history.append(np.mean(episode_losses))
        else:
            losses_history.append(0.0)

        corinthians_bot.update_target()
        palmeiras_bot.update_target()
        corinthians_bot.epsilon = max(0.1, corinthians_bot.epsilon * 0.99)
        palmeiras_bot.epsilon = max(0.1, palmeiras_bot.epsilon * 0.99)
        print(f"Episódio {episode+1} concluído. Vencedor: {episode_winner}. Epsilon Corinthians: {corinthians_bot.epsilon:.2f}, Palmeiras: {palmeiras_bot.epsilon:.2f}")

    corinthians_bot.save_model(os.path.join(MODEL_DIR, "bot_corinthians_model.pth"))
    palmeiras_bot.save_model(os.path.join(MODEL_DIR, "bot_palmeiras_model.pth"))
    plot_performance(rewards_history, losses_history, "training_performance.png")
    plot_game_results(game_results_history, "training_game_results.png")
    pygame.quit()


# --- Player vs AI Mode ---
def player_vs_ai_mode():
    pygame.init()
    # Carregar imagens uma vez no início do modo
    load_unit_images() # CHAMADA AQUI!
    screen_width = BOARD_SIZE * TILE_SIZE
    screen_height = BOARD_SIZE * TILE_SIZE
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Jogador vs IA - Corinthians")

    input_dim = UNITS_PER_BOT * 3 * 2
    action_dim = UNITS_PER_BOT * 5

    player_bot = Bot("Jogador", (0, 150, 0), input_dim, action_dim)
    ai_bot = Bot("Corinthians", (255, 0, 0), input_dim, action_dim)

    if not ai_bot.load_model(os.path.join(MODEL_DIR, "bot_corinthians_model.pth")):
        print("Modelo da IA (Corinthians) não encontrado. A IA jogará aleatoriamente no início.")
    ai_bot.epsilon = 0.1

    place_units_strategically(player_bot, ai_bot, BOARD_SIZE)
    MESSAGE_LOG.clear()
    
    current_player_turn = True
    game_over = False

    selected_unit = None

    while not game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.MOUSEBUTTONDOWN and current_player_turn and not game_over:
                mouse_x, mouse_y = event.pos
                clicked_col = mouse_x // TILE_SIZE
                clicked_row = mouse_y // TILE_SIZE

                unit_found = False
                for i, unit in enumerate(player_bot.get_alive_units()):
                    if unit.x == clicked_col and unit.y == clicked_row:
                        selected_unit = unit
                        player_bot.send_message(f"Unidade {i+1} selecionada.")
                        unit_found = True
                        break
                
                if not unit_found and selected_unit:
                    possible_move = False
                    action_taken = -1
                    
                    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
                    for i, (dx, dy) in enumerate(directions):
                        if selected_unit.x + dx == clicked_col and selected_unit.y + dy == clicked_row:
                            action_taken = i
                            possible_move = True
                            break
                    
                    if not possible_move:
                        for enemy_unit in ai_bot.get_alive_units():
                            if enemy_unit.x == clicked_col and enemy_unit.y == clicked_row and \
                                abs(selected_unit.x - enemy_unit.x) + abs(selected_unit.y - enemy_unit.y) == 1:
                                action_taken = 4
                                possible_move = True
                                break

                    if possible_move:
                        unit_idx = player_bot.get_alive_units().index(selected_unit)
                        player_action = unit_idx * 5 + action_taken
                        
                        game_over, reward, msg = make_move(player_bot, ai_bot, is_player_turn=True, player_action=player_action)
                        selected_unit = None
                        if not game_over:
                            current_player_turn = False

        if not current_player_turn and not game_over:
            game_over, reward, msg = make_move(ai_bot, player_bot, is_player_turn=False)
            if not game_over:
                current_player_turn = True

        draw_board(screen, player_bot, ai_bot, "Sua vez" if current_player_turn else "Vez da IA")
        update_web_game_state(player_bot, ai_bot, "Sua vez" if current_player_turn else "Vez da IA", game_over, "Jogador" if (game_over and len(player_bot.get_alive_units()) > 0) else (ai_bot.name if (game_over and len(ai_bot.get_alive_units()) > 0) else None))

        if game_over:
            if not player_bot.get_alive_units():
                ai_bot.send_message("Eu venci!")
                print(f"A IA ({ai_bot.name}) venceu!")
                update_web_game_state(player_bot, ai_bot, f"A IA ({ai_bot.name}) venceu!", True, ai_bot.name)
            elif not ai_bot.get_alive_units():
                player_bot.send_message("Eu venci!")
                print("Você venceu!")
                update_web_game_state(player_bot, ai_bot, "Você venceu!", True, "Jogador")
            else:
                player_bot.send_message("Fim de jogo!")
                ai_bot.send_message("Fim de jogo!")
                print("Fim de jogo!")
            pygame.time.wait(3000)
            break
        
    pygame.quit()


# --- Main Menu ---
def main_menu():
    pygame.init()
    screen_width = 600
    screen_height = 400
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Menu Principal")
    font = pygame.font.Font(None, 40)

    white = (255, 255, 255)
    blue = (0, 0, 255)
    gray = (200, 200, 200)
    black = (0, 0, 0)

    button_width = 500
    button_height = 60
    
    button_x = (screen_width - button_width) // 2

    train_button_rect = pygame.Rect(button_x, 80, button_width, button_height)
    player_vs_ai_button_rect = pygame.Rect(button_x, 180, button_width, button_height)
    web_interface_button_rect = pygame.Rect(button_x, 280, button_width, button_height) if FLASK_AVAILABLE else None

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                if train_button_rect.collidepoint(event.pos):
                    train_mode()
                    running = False
                elif player_vs_ai_button_rect.collidepoint(event.pos):
                    player_vs_ai_mode()
                    running = False
                elif web_interface_button_rect and web_interface_button_rect.collidepoint(event.pos):
                    print("Iniciando interface web em http://127.0.0.1:5000")
                    if FLASK_AVAILABLE:
                        web_thread = Thread(target=run_flask_app)
                        web_thread.daemon = True
                        web_thread.start()
                        pygame.quit()
                        return
                    else:
                        print("Flask não está disponível. Não é possível iniciar a interface web.")
        
        screen.fill(white)

        pygame.draw.rect(screen, blue, train_button_rect)
        train_text = font.render("Modo Treinamento (Corinthians vs Palmeiras)", True, white)
        train_text_rect = train_text.get_rect(center=train_button_rect.center)
        screen.blit(train_text, train_text_rect)

        pygame.draw.rect(screen, blue, player_vs_ai_button_rect)
        player_vs_ai_text = font.render("Modo Jogador vs Corinthians", True, white)
        player_vs_ai_text_rect = player_vs_ai_text.get_rect(center=player_vs_ai_button_rect.center)
        screen.blit(player_vs_ai_text, player_vs_ai_text_rect)

        if web_interface_button_rect:
            pygame.draw.rect(screen, blue, web_interface_button_rect)
            web_text = font.render("Iniciar Interface Web (Opcional)", True, white)
            web_text_rect = web_text.get_rect(center=web_interface_button_rect.center)
            screen.blit(web_text, web_text_rect)
        else:
            disabled_web_button_rect = pygame.Rect(button_x, 280, button_width, button_height)
            pygame.draw.rect(screen, gray, disabled_web_button_rect)
            disabled_web_text = font.render("Flask não instalado (Web Inativo)", True, black)
            disabled_web_text_rect = disabled_web_text.get_rect(center=disabled_web_button_rect.center)
            screen.blit(disabled_web_text, disabled_web_text_rect)


        pygame.display.flip()

    pygame.quit()

# --- Entry Point ---
if __name__ == '__main__':
    main_menu()