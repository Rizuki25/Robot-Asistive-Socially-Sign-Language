import cors from "cors";
import express from "express";
import http from "http";
import { Server } from "socket.io";

const PORT = process.env.PORT || 3001;
const CLIENT_ORIGIN = process.env.CLIENT_ORIGIN || "*";
const app = express();
const server = http.createServer(app);

app.use(
  cors({
    origin: CLIENT_ORIGIN,
  }),
);
app.use(express.json());

const io = new Server(server, {
  cors: {
    origin: CLIENT_ORIGIN,
  },
});

app.get("/health", (_request, response) => {
  response.json({ status: "ok" });
});

app.post("/api/sign-result", (request, response) => {
  const { roomId = "demo-ta", text } = request.body;

  if (!text || typeof text !== "string") {
    response.status(400).json({ message: "Field text wajib diisi." });
    return;
  }

  const message = {
    roomId,
    sender: "sign",
    text: text.trim(),
    timestamp: Date.now(),
  };

  io.to(roomId).emit("message", message);
  response.json({ message: "Hasil bahasa isyarat dikirim.", data: message });
});

io.on("connection", (socket) => {
  console.log("Client connected:", socket.id);

  socket.on("join-room", (roomId) => {
    socket.join(roomId);
    console.log(`${socket.id} joined room ${roomId}`);
  });

  socket.on("send-message", (message) => {
    const nextMessage = {
      roomId: message.roomId || "demo-ta",
      sender: message.sender || "system",
      text: String(message.text || "").trim(),
      timestamp: Date.now(),
    };

    if (!nextMessage.text) {
      return;
    }

    io.to(nextMessage.roomId).emit("message", nextMessage);
  });

  socket.on("disconnect", () => {
    console.log("Client disconnected:", socket.id);
  });
});

server.listen(PORT, "0.0.0.0", () => {
  console.log(`Socket.IO server running on http://0.0.0.0:${PORT}`);
});
