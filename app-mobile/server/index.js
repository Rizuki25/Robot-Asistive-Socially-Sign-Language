import cors from "cors";
import express from "express";
import http from "http";
import { Server } from "socket.io";

const PORT = process.env.PORT || 3001;
const CLIENT_ORIGIN = process.env.CLIENT_ORIGIN || "*";
const MODEL_API_KEY = process.env.MODEL_API_KEY || "";
const app = express();
const server = http.createServer(app);

function requireModelApiKey(request, response, next) {
  if (!MODEL_API_KEY) {
    next();
    return;
  }

  if (request.get("x-model-api-key") !== MODEL_API_KEY) {
    response.status(401).json({ message: "API key model tidak valid." });
    return;
  }

  next();
}

function getRoomId(request) {
  return String(request.query.roomId || request.get("x-room-id") || "demo-ta")
    .trim()
    .slice(0, 100) || "demo-ta";
}

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

app.post("/api/sign-result", requireModelApiKey, (request, response) => {
  const {
    roomId = "demo-ta",
    text,
    confidence = null,
    source = "model",
  } = request.body;

  if (!text || typeof text !== "string") {
    response.status(400).json({ message: "Field text wajib diisi." });
    return;
  }

  const message = {
    roomId,
    sender: "sign",
    text: text.trim(),
    timestamp: Date.now(),
    confidence:
      typeof confidence === "number" && Number.isFinite(confidence)
        ? confidence
        : null,
    source: String(source || "model"),
  };

  io.to(roomId).emit("message", message);
  response.json({ message: "Hasil bahasa isyarat dikirim.", data: message });
});

app.post(
  "/api/video-frame",
  requireModelApiKey,
  express.raw({ type: "image/jpeg", limit: "2mb" }),
  (request, response) => {
    if (!Buffer.isBuffer(request.body) || request.body.length === 0) {
      response.status(400).json({ message: "Frame JPEG wajib dikirim." });
      return;
    }

    const roomId = getRoomId(request);
    io.to(roomId).emit("video-frame", {
      roomId,
      image: request.body,
      timestamp: Date.now(),
    });
    response.status(202).end();
  },
);

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
