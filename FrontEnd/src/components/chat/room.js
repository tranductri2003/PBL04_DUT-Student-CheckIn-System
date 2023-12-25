import React, { Component } from "react";
import { w3cwebsocket as W3CWebSocket } from "websocket";
import jwt_decode from "jwt-decode";

import Button from "@material-ui/core/Button";
import CssBaseline from "@material-ui/core/CssBaseline";
import TextField from "@material-ui/core/TextField";
import Container from "@material-ui/core/Container";
import Paper from "@material-ui/core/Paper";
import Typography from "@material-ui/core/Typography";
import Avatar from "@material-ui/core/Avatar";

import { withStyles } from "@material-ui/core/styles";
import "./room.css";


class ChatApp extends Component {
    constructor(props) {
        super(props);

        const token = localStorage.getItem("access_token"); // Thay thế bằng cách lấy token từ nơi bạn lưu trữ nó

        // Giải mã token
        const decodedToken = jwt_decode(token);
        this.state = {
            filledForm: false,
            messages: this.props.messages || [], // Sử dụng messages từ props nếu có, nếu không thì mặc định là mảng rỗng
            content: "",
            staff_id: decodedToken.staff_id,
            room_slug: this.props.room_slug || "nang-bach-khoa",
            room_title: this.props.room_title || "Nẵng Bách Khoa",
        };
        /* This line of code is creating a new instance of the `W3CWebSocket` class and initializing it
        with the WebSocket URL. The URL is constructed using the room name from the component's
        state. The WebSocket URL is used to establish a connection between the client and the server
        for real-time communication. */
        // this.client = new W3CWebSocket("ws://127.0.0.1:8000/ws/" + this.state.room_slug + "/");
        const websocketURL = `${process.env.REACT_APP_CHAT_WEBSOCKET_URL}${this.state.room_slug}/`;
        this.client = new W3CWebSocket(websocketURL);

    }
    onButtonClicked = (e) => {
        // Tạo đối tượng messageData từ state
        const messageData = {
            type: "message",
            message: this.state.content,
            staff_id: this.state.staff_id,
            room_slug: this.state.room_slug,
        };

        // In ra dòng mã JSON trước khi gửi nó
        console.log("Message Data:", JSON.stringify(messageData));

        // Gửi tin nhắn đến máy chủ WebSocket
        this.client.send(JSON.stringify(messageData));

        // Đặt lại trạng thái content
        this.setState({ content: "" });

        e.preventDefault();
    };


    componentDidMount() {
        this.client.onopen = () => {
            console.log("WebSocket Client Connected");
            const token = localStorage.getItem("access_token");
            this.client.send(JSON.stringify({ "access_token": token })); // Gửi access_token ngay khi kết nối WebSocket được thiết lập
            console.log("Sent access_token as first message!");
        };
        this.client.onmessage = (message) => {
            const dataFromServer = JSON.parse(message.data);
            if (dataFromServer) {
                console.log("RECEIVE DATA FROM SERVER", dataFromServer);
                this.setState((state) => ({
                    messages: [
                        ...state.messages,
                        {
                            message: dataFromServer.message,
                            staff_id: dataFromServer.staff_id,
                        },
                    ],
                }));
                console.log(this.state.messages);
                this.scrollToBottom();
            }
        };
    }

    componentDidUpdate() {
        this.scrollToBottom();
    }

    scrollToBottom() {
        const chatboxContainer = document.getElementById("chatbox-container");
        chatboxContainer.scrollTop = chatboxContainer.scrollHeight;
    }

    render() {
        const { classes } = this.props;
        return (
            <Container component="main" maxWidth="md">
                <CssBaseline />
                <div className="root">
                    <div className="titleContainer">
                        <Typography variant="h6" color="textPrimary" gutterBottom>
                            <span role="img" aria-label="Chat Icon">
                                💬
                            </span>{" "}
                            Chat Room: {this.state.room_slug}
                        </Typography>
                    </div>
                    <div className="chatBoxContainer" id="chatbox-container">
                        <Paper className="chatBox">
                            {this.state.messages.length > 0 ? (
                                this.state.messages.map((message, index) => (
                                    <div
                                        key={index}
                                        className={`message ${message.staff_id === this.state.staff_id ? 'messageRight' : 'messageLeft'
                                            }`}
                                    >
                                        <Avatar className="avatar">
                                            {message.staff_id.charAt(0).toUpperCase()}
                                        </Avatar>
                                        <div className="messageText">
                                            <Typography variant="subtitle2" className="messageSender">
                                                {message.staff_id}
                                            </Typography>
                                            <Typography variant="body1">{message.message}</Typography>
                                        </div>
                                    </div>
                                ))
                            ) : (
                                <Typography variant="body1">No Messages in this Room.</Typography>
                            )}
                        </Paper>
                    </div>
                    <form className="form" noValidate onSubmit={this.onButtonClicked}>
                        <TextField
                            className="input"
                            placeholder="Enter text here"
                            variant="outlined"
                            fullWidth
                            value={this.state.content}
                            onChange={(e) => this.setState({ content: e.target.value })}
                        />
                        <Button
                            type="submit"
                            variant="contained"
                            color="primary"
                            className="submit"
                            disableElevation
                        >
                            Send Message
                        </Button>
                    </form>
                </div>
            </Container>
        );
    }
}
export default ChatApp;
