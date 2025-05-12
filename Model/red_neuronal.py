import tensorflow as tf
import numpy as np
X = 1
O = -1
empty = 0

""" 
def board_to_state(board):
    return np.reshape(board, (9,))
def make_move(board, position, player):
    board[position] = player
    return board
def get_possible_moves(board):
    return np.where(board == empty)[0]
def create_training_data():
    X_train = []
    y_train = []

    for _ in range(10000):

        board = np.random.choice([X, O, empty], size=9, p=[1/3, 1/3, 1/3])
        while np.all(board != empty):
            board = np.random.choice([X, O, empty], size=9, p=[1/3, 1/3, 1/3])

        possible_moves = get_possible_moves(board)
        optimal_move = np.random.choice(possible_moves) if len(possible_moves) > 0 else None

        if optimal_move is not None:
            X_train.append(board_to_state(board))
            y_train.append(optimal_move)

    return np.array(X_train), np.array(y_train)


# Definir la red neuronal
def create_model():

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(9,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(9, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    X_train, y_train = create_training_data()
    model.fit(X_train, y_train, epochs=100,verbose=0)
    return model


def imprimir_tablero(tablero):
    symbols = {
        empty: "⬜",
        X: "❌",
        O: "🧿"
    }

    for i in range(3):
        print("".join(symbols[tablero[i * 3 + j]] for j in range(3)))

def check_winner(board):
    for i in range(3):
        if board[i*3] == board[i*3+1] == board[i*3+2] != empty:
            return board[i*3]

    for i in range(3):
        if board[i] == board[i+3] == board[i+6] != empty:
            return board[i]

    if board[0] == board[4] == board[8] != empty:
        return board[0]
    if board[2] == board[4] == board[6] != empty:
        return board[2]

    return None

def check_draw(board):
    return np.all(board != empty) and check_winner(board) is None

def play(model):
    board = np.zeros(9)

    while True:
        # Movimiento de la IA
        state = board_to_state(board)
        prediction = model.predict(state.reshape(1, -1))
        machine_move = np.argmax(prediction)

        while board[machine_move] != empty:
            prediction = model.predict(state.reshape(1, -1))
            machine_move = np.argmax(prediction)

        board = make_move(board, machine_move, X)
        imprimir_tablero(board)

        if check_winner(board) == X:
            print("La IA ha ganado!")
            break
        elif check_draw(board):
            print("Empate.")
            break

        # Movimiento del jugador
        while True:
            try:
                human_move = int(input("Tu movimiento (0-8): "))
                if human_move < 0 or human_move > 8 or board[human_move] != empty:
                    raise ValueError("Movimiento no válido. Intenta otra vez.")
                break
            except ValueError as e:
                print(e)

        board = make_move(board, human_move, O)
        imprimir_tablero(board)

        if check_winner(board) == O:
            print("¡Has ganado!")
            break
        elif check_draw(board):
            print("Empate.")
            break
"""

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu',input_shape=(9,)),
        tf.keras.layers.LeakyReLU(alpha=0.01),
        tf.keras.layers.Dense(64),
        tf.keras.layers.LeakyReLU(alpha=0.01),
        tf.keras.layers.Dense(32),
        tf.keras.layers.LeakyReLU(alpha=0.01),
        tf.keras.layers.Dense(9, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), 
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def make_move(board, model):
    state = np.reshape(board, (1, 9))
    prediction = model.predict(state)
    machine_move = np.argmax(prediction)

    while board[machine_move]!=empty:
        prediction[0,machine_move]= -np.inf
        machine_move = np.argmax(prediction)

    return machine_move

def check_winner(board):

    win_conditions = [
        (0, 1, 2), 
        (3, 4, 5), 
        (6, 7, 8), 
        (0, 3, 6), 
        (1, 4, 7), 
        (2, 5, 8), 
        (0, 4, 8), 
        (2, 4, 6)
    ]
    
    for x, y, z in win_conditions:
        if board[x] == board[y] == board[z] != 0:
            return board[x]
    return 0

def check_draw(board):
    if np.all(board != 0) and check_winner(board) == 0:
        return True
    return False

def generate_training_data(num_games=10000):
    inputs = []
    outputs = []

    for _ in range(num_games):
        board = np.zeros(9)  
        moves = []

        for turn in range(9):
            available_positions = np.where(board == 0)[0]
            move = np.random.choice(available_positions)
            board[move] = 1 if turn % 2 == 0 else -1
            moves.append((board.copy(), move))
            if check_winner(board):
                break

        for board_state, move_position in moves:
            inputs.append(board_state)
            output = np.zeros(9)
            output[move_position] = 1
            outputs.append(output)

    return np.array(inputs), np.array(outputs)

def get_ai_move(board,model):
    
    available_moves = [i for i, x in enumerate(board) if x == 0]  
    
    if not available_moves:  
        return -1  

    predictions = model.predict(np.array([board]))  
    best_move = available_moves[np.argmax(predictions[0][available_moves])]
    
    return best_move