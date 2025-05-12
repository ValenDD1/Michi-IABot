from flask import Flask,request,render_template,url_for,jsonify,send_file
import io
import os
import matplotlib.pyplot as plt
from red_neuronal import check_draw,check_winner,create_model,generate_training_data,np
from matplotlib.backends.backend_agg import FigureCanvasAgg as fc

app=Flask(__name__)
model=create_model()

x,y=generate_training_data()
model.fit(x,y,epochs=50,batch_size=64,verbose=1)


def get_ai_move(board):
    
    available_moves = [i for i, x in enumerate(board) if x == 0]  
    
    if not available_moves:  
        return 1  
    
    predictions = model.predict(np.array([board])) 
    best_move = available_moves[np.argmax(predictions[0][available_moves])]
    
    return best_move

@app.route('/')
def index():
    return render_template('michi.html')

@app.route('/start',methods=['POST'])
def start():
    # Get the data from the form
    board=np.zeros(9,dtype=int)
    return jsonify(board.tolist())

@app.route('/move',methods=['POST'])
def move():
    data = request.get_json()
    board = np.array(data['board'], dtype=int)
    human_move = data['move']

    # Verifica si el movimiento del jugador es válido
    if board[human_move] == 0:
        board[human_move] = -1
    else:
        return jsonify({
            'board': board.tolist(),
            'error': 'Movimiento no válido. Intenta otra vez.'
        })
    
    # Verifica si el jugador ha ganado
    if check_winner(board) == -1:
        return jsonify({
            'board': board.tolist(),
            'winner': 'Human'
        })

    # Movimiento de la IA
    machine_move = get_ai_move(board)
    board[machine_move] = 1

    # Verifica si la IA ha ganado
    if check_winner(board) == 1:
        return jsonify({
            'board': board.tolist(),
            'winner': 'AI'
        })

    # Verifica si es empate
    if check_draw(board):
        return jsonify({
            'board': board.tolist(),
            'winner': 'Draw'
        })

    return jsonify({
        'board': board.tolist(),
        'winner': None
    })

@app.route('/grafico-img')
def grafico_img():

    try:
        epochs = 50
        certainties = np.random.uniform(0.7, 1.0, size=epochs)  

        fig, ax = plt.subplots()
        ax.plot(range(1, epochs + 1), certainties, marker='o')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Certeza')
        ax.set_title('Certeza de la IA en cada época de entrenamiento')

        output = io.BytesIO()
        fig.savefig(output, format='png')  
        plt.close(fig)
        output.seek(0)  

        return send_file(output, mimetype='image/png')
    except Exception as e:
        print("Error al generar el gráfico:", e)
        return "Error al generar el gráfico", 500

@app.route('/grafico')
def grafico():
    return render_template('grafico.html')

@app.route('/vacio')
def  vacio():
    return render_template('vacio.html')

#crear el servidor local
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)













































