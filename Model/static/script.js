let board = [];
let playerSymbol = "O";
let aiSymbol = "X";
let gameEnded = false;  
let xWins = 0;
let oWins = 0;

async function startGame() {
    gameEnded = false;
    if (!gameEnded) {  
        const response = await fetch("/start", { method: "POST" });
        if (response.ok) {
            board = await response.json();
            renderBoard();
            document.getElementById("status").innerText = "";
            document.getElementById("iniciar-partida").style.display = "none";
            
        } else {
            console.error("Error al iniciar el juego.");
            document.getElementById("status").innerText = "Error al iniciar el juego.";
        }
    }
}

function renderBoard() {
    const boardDiv = document.getElementById("tablero-michi");
    boardDiv.innerHTML = "";

    board.forEach((cell, index) => {
        const cellDiv = document.createElement("div");
        cellDiv.className = "cell";
        cellDiv.innerText = cell === 1 ? aiSymbol : cell === -1 ? playerSymbol : "";

        if (gameEnded) {
            cellDiv.style.pointerEvents = "none"; 
        } else {
            cellDiv.style.pointerEvents = "auto"; 
        }
        if (cell === 0 && !gameEnded) {
            cellDiv.onclick = () => makeMove(index);
        }

        boardDiv.appendChild(cellDiv);
    });
}

async function makeMove(index) {

    if (board[index] !== 0 || gameEnded) return;

    const response = await fetch("/move", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ board: board, move: index })
    });

    if (response.ok) {
        const data = await response.json();
        board = data.board;
        renderBoard();

        if (data.winner) {
            if (data.winner === "Human") {
                document.getElementById("status").innerText = "¡Ganaste!";
                document.getElementById("grafico").style.display = "block";
                xWins += 1;
            } else if (data.winner === "AI") {
                document.getElementById("status").innerText = "La IA ganó.";
                document.getElementById("grafico").style.display = "block";
                oWins += 1;
            } else {
                document.getElementById("status").innerText = "Empate.";
                document.getElementById("grafico").style.display = "block";
            }
            gameEnded = true; 
            document.getElementById("iniciar-partida").style.display = "block";
            
        }
        console.log(data.winner)
    } else {
        console.error("Error en el movimiento.");
        console.log(data.winner)
        document.getElementById("status").innerText = "Error en el movimiento.";
    }
    document.getElementById('puntaje-ia').innerText = oWins;
    document.getElementById('puntaje-jugador').innerText = xWins;
}

document.getElementById("iniciar-partida").addEventListener("click", startGame);

document.getElementById('puntaje-ia').innerText = oWins;
document.getElementById('puntaje-jugador').innerText = xWins;