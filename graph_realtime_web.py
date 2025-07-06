from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import os

app = Flask(__name__, static_folder="output", template_folder="templates")
socketio = SocketIO(app, cors_allowed_origins='*')  # CORS を許可（同一 PC 内ブラウザ想定）

GRAPH_PATH = "output/relationship_graph.png"


# ====== HTTP ======
@app.route("/")
def index():
    return render_template("index.html")


# ====== Socket.IO ======
@socketio.on("control")           # ブラウザ ▶ / ⏸ ボタン → サーバへ
def handle_control(data):
    """
    data = {"action": "start"}  または  {"action": "stop"}
    同じイベントをそのまま Python クライアント（realtime_communicator）にも中継
    """
    emit("control", data, broadcast=True, include_self=False)


@socketio.on("graph_updated")     # Python クライアント → サーバへ
def relay_graph_event():
    """
    グラフ画像が保存されたら Python クライアントが emit してくる
    ブラウザへリレー → JS 側で <img> の src パラメータを付け替えてリロード
    """
    emit("refresh_graph", broadcast=True)


@socketio.on("conversation_update")  # Python クライアント → サーバへ
def relay_conversation(msg):
    """
    msg = {"speaker": "小野寺", "utterance": "こんにちは"}
    """
    emit("conversation_update", msg, broadcast=True)


@socketio.on("robot_speak")  # Pythonクライアント → サーバへ
def relay_robot_speak(msg):
    """
    msg = {"speaker": "ロボット", "utterance": "〇〇さん、どう思いますか？"}
    ブラウザにロボットの発言を中継
    """
    emit("conversation_update", msg, broadcast=True)
    emit("robot_log", msg, broadcast=True)


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8888))
    socketio.run(app, port=port, debug=True)
