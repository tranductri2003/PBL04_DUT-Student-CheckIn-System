import React, { useEffect, useState, useRef } from 'react';
import axiosInstance from '../../axios';
import { w3cwebsocket as W3CWebSocket } from 'websocket';
import { notification } from 'antd';
import Webcam from 'react-webcam';
import styled from 'styled-components';
import download from 'downloadjs';
import { FaCheck } from 'react-icons/fa'; // Sử dụng thư viện react-icons cho biểu tượng tick


let isFaceValidated = true; // Global variable for overall validation
let isLocationValidated = true;


const WebcamOverlay = styled.div`
  position: absolute;
  top: 0;
  left: 18%;
  width: 65%;
  height: 80%;
  border: 2px solid #4caf50; // Viền nổi bật cho khu vực webcam
  padding: 10px; // Đệm xung quanh webcam

  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.5); // Thêm bóng đổ
  border-radius: 10px; // Bo tròn góc
  overflow: hidden; // Ẩn các phần tử vượt ra ngoài viền
  display: flex; // Sử dụng flexbox để căn chỉnh các thành phần bên trong
  justify-content: center; // Căn giữa các thành phần theo trục ngang
  align-items: center; // Căn giữa các thành phần theo trục dọc
  background-color: #fff; // Màu nền tùy chọn
`;

const CaptureButton = styled.button`
  position: absolute;
  bottom: 0px;
  left: 50%;
  transform: translateX(-50%);
  background-color: #4caf50;
  color: white;
  padding: 10px 20px;
  border: none;
  border-radius: 5px;
  cursor: pointer;
  fontWeight: 'bold', // Chữ in đậm

`;

const ButtonContainer = styled.div`
  position: absolute;
  bottom: 10px;
  left: 10px;
  display: flex;
  gap: 10px; // Khoảng cách giữa các nút
  fontWeight: 'bold', // Chữ in đậm

`;
const modalStyles = {

    AttendanceModal: {
        fontFamily: '"Helvetica Neue", Arial, sans-serif',
        textAlign: 'center',
        margin: '20px',
    },
    tableContainer: {
        width: '100%',
        overflowX: 'auto',
        boxShadow: '0 4px 8px 0 rgba(0,0,0,0.2)', // Thêm bóng đổ

    },
    tableWrapper: {
        maxHeight: 'calc(90vh - 120px)',
        overflowY: 'auto',
        backgroundColor: '#f8f9fa', // Nền nhẹ cho bao bọc

    },
    table: {
        width: '100%',
        borderCollapse: 'collapse',
        border: '1px solid #ddd',
        margin: '0 auto',
        backgroundColor: '#ffffff', // Nền trắng cho bảng

    },
    tableHeader: {
        padding: '16px',
        backgroundColor: 'darkblue', // Sử dụng màu xanh cho tiêu đề
        color: 'white', // Màu chữ trắng cho đối lập
        fontSize: '18px',
        fontWeight: 'bold',
        position: 'sticky',
        top: '0',
        fontFamily: '"Helvetica Neue", Arial, sans-serif', // Updated to a more sophisticated font

    },
    cell: {
        fontFamily: '"Helvetica Neue", Arial, sans-serif', // Updated to a more sophisticated font
        fontWeight: 'bold', // Chữ in đậm
        padding: '12px',
        textAlign: 'center',
        fontSize: '16px',
        borderBottom: '1px solid #ddd', // Thêm viền cho mỗi ô
        '&:hover': {
            backgroundColor: '#f2f2f2', // Thêm hiệu ứng di chuột
        },

    },
    link: {
        textDecoration: 'none',
        color: 'blue',
        cursor: 'pointer',
    },
    tick: {
        fontSize: '20px',
        color: 'green',
    },
    cross: {
        fontSize: '20px',
        color: 'red',
    },
    cameraButton: {
        backgroundColor: 'green',
        color: 'white',
        border: 'none',
        cursor: 'pointer',
        padding: '10px 20px',
        borderRadius: '5px',
        fontWeight: 'bold', // Chữ in đậm

    },
    checkLocationButton: {
        backgroundColor: 'green',
        color: 'white',
        border: 'none',
        cursor: 'pointer',
        padding: '10px 20px',
        borderRadius: '5px',
        fontWeight: 'bold', // Chữ in đậm

    },
    button: {
        position: 'absolute',
        bottom: '10px', // Adjust top position as needed
        backgroundColor: 'green',
        color: 'white',
        border: 'none',
        cursor: 'pointer',
        padding: '10px 20px',
        borderRadius: '5px',
        margin: "center",
        fontWeight: 'bold', // Chữ in đậm

    },
};


function isWithinTimeRange(startTime, endTime) {
    if (startTime && endTime) {
        const currentTime = new Date();
        const currentHours = currentTime.getHours();
        const currentMinutes = currentTime.getMinutes();
        const [startHour, startMinute] = startTime.split(':').map(Number);
        const [endHour, endMinute] = endTime.split(':').map(Number);

        if (
            currentHours > startHour ||
            (currentHours === startHour && currentMinutes >= startMinute)
        ) {
            if (
                currentHours < endHour ||
                (currentHours === endHour && currentMinutes < endMinute)
            ) {
                return true;
            }
        }
    }
    return false;
}


const AttendanceModal = (props) => {
    const { course } = props;

    const [users, setUsers] = useState([]);
    const [presentStudents, setPresentStudents] = useState([]);
    const [websocket, setWebsocket] = useState(null);

    const [webcamVisible, setWebcamVisible] = useState(false);
    const [capturedImage, setCapturedImage] = useState(null);
    const webcamRef = useRef(null);

    const [savedPosition, setSavedPosition] = useState(null);
    const [avatars, setAvatars] = useState({});

    // const [isImageCaptured, setIsImageCaptured] = useState(false);
    // const [isLocationValidated, setisLocationValidated] = useState(false);


    const saveImageToLocalstorage = (imageName, imageData) => {
        localStorage.setItem(imageName, imageData);
    };



    // Function to convert data URI to Blob
    function dataURItoBlob(dataURI) {
        const byteString = atob(dataURI.split(',')[1]);
        const mimeString = dataURI.split(',')[0].split(':')[1].split(';')[0];
        const ab = new ArrayBuffer(byteString.length);
        const ia = new Uint8Array(ab);
        for (let i = 0; i < byteString.length; i++) {
            ia[i] = byteString.charCodeAt(i);
        }
        return new Blob([ab], { type: mimeString });
    }




    const handleCapture = () => {
        const image = webcamRef.current.getScreenshot();
        setCapturedImage(image);
        setWebcamVisible(false);
        saveImageToLocalstorage('check_in_image', image);
        // setIsImageCaptured(true);
        // console.log('isImageCaptured:', isImageCaptured); // Add this line to log the state

        const token = localStorage.getItem('access_token');

        const formData = new FormData();
        formData.append('staff_id', '102210096');
        formData.append('image', dataURItoBlob(image)); // Convert data URI to Blob

        console.log(isFaceValidated);
        axiosInstance.post(
            `${process.env.REACT_APP_AI_URL}`,
            formData,
            {
                headers: {
                    'Content-Type': 'multipart/form-data',
                    'Authorization': `Bearer ${token}`,
                },
            }
        )
            .then(response => {
                isFaceValidated = response.data.validated; // Update the face validation status
                console.log(response.data);
                console.log(isFaceValidated);

            })
            .catch(error => {
                console.error('Lỗi khi tải dữ liệu users:', error);
            });
    };



    // Sử dụng useEffect để thiết lập kết nối WebSocket khi trang được tải
    useEffect(() => {
        const token = localStorage.getItem('access_token');
        const websocketURL = `${process.env.REACT_APP_CHECK_IN_WEBSOCKET_URL}${course.course_id}/`;
        const client = new W3CWebSocket(websocketURL);

        client.onopen = () => {
            console.log('WebSocket Client Connected');
            client.send(JSON.stringify({ 'access_token': token }));
            console.log('Sent access_token as the first message!');
        };

        client.onmessage = (message) => {
            const dataFromServer = JSON.parse(message.data);
            if (dataFromServer) {
                console.log('RECEIVE DATA FROM SERVER', dataFromServer);
                console.log(dataFromServer.message)
                // Lấy student_id từ dữ liệu nhận được

                const studentId = dataFromServer.message.student_id;

                // Cập nhật presentStudents
                setPresentStudents((prevPresentStudents) => {
                    const updatedPresentStudents = { ...prevPresentStudents };

                    // Kiểm tra nếu student_id đã tồn tại trong danh sách presentStudents                        // Cập nhật trạng thái điểm danh của sinh viên
                    updatedPresentStudents[studentId] = true; // Ví dụ: Đánh dấu là đã điểm danh
                    return updatedPresentStudents;
                });

                notification.success({
                    message: dataFromServer.message,
                    description: 'Congratulations!!!',
                    placement: 'topRight'
                });
            }
        };

        setWebsocket(client);

        // Loại bỏ kết nối WebSocket khi Component unmount
        return () => {
            if (websocket) {
                websocket.close();
            }
        };
    }, []); // Rỗng [] đảm bảo hiệu ứng này chỉ chạy một lần khi trang được tải.

    function calculateDistance(lat1, lon1, lat2, lon2) {
        const R = 6371; // Bán kính trái đất trong km
        const dLat = (lat2 - lat1) * (Math.PI / 180);
        const dLon = (lon2 - lon1) * (Math.PI / 180);
        const a =
            Math.sin(dLat / 2) * Math.sin(dLat / 2) +
            Math.cos(lat1 * (Math.PI / 180)) *
            Math.cos(lat2 * (Math.PI / 180)) *
            Math.sin(dLon / 2) *
            Math.sin(dLon / 2);
        const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
        const distance = R * c; // Khoảng cách trong km
        return distance;
    }

    const handleCheckLocation = () => {
        if (navigator.geolocation) {
            navigator.geolocation.getCurrentPosition((position) => {
                const { latitude, longitude } = position.coords;
                const currentPosition = { latitude, longitude };
                localStorage.setItem('position', JSON.stringify(currentPosition));

                // Show the coordinates on the screen
                notification.info({
                    message: 'Current Location',
                    description: `Latitude: ${latitude}, Longitude: ${longitude}`,
                    placement: 'topRight'
                });

                // Open Google Maps with the current coordinates
                const googleMapsLink = `https://www.google.com/maps?q=${latitude},${longitude}`;
                // window.open(googleMapsLink, '_blank');

                // Set the saved position to state for rendering on the screen
                setSavedPosition(currentPosition);

                const distance = calculateDistance(
                    latitude,
                    longitude,
                    16.073981,
                    108.149891
                );
                console.log(isLocationValidated);
                if (distance <= 1) {
                    // Nếu khoảng cách lớn hơn 1km, thông báo
                    isLocationValidated = true;
                    notification.success({
                        message: 'Distance Check',
                        description: `The distance is smaller than 1km. Goood.`,
                        placement: 'topRight'
                    });

                } else {
                    isLocationValidated = true;
                    notification.error({
                        message: 'Distance Check',
                        description: `The distance is greater than 1km. Please check in within 1km.`,
                        placement: 'topRight'
                    });
                    console.log('isLocationValidated:', isLocationValidated); // Add this line to log the state

                }
                console.log(isLocationValidated);




            }, (error) => {
                console.error('Error getting location:', error);
                notification.error({
                    message: 'Error',
                    description: 'Could not retrieve location.',
                    placement: 'topRight'
                });
            });
        } else {
            // Geolocation is not supported
            // Handle the case where geolocation is not supported
            notification.error({
                message: 'Error',
                description: 'Geolocation is not supported by your browser.',
                placement: 'topRight'
            });
        }
    };



    // Hàm gửi dữ liệu thông qua WebSocket
    const sendWebSocketData = () => {
        if (isFaceValidated && isLocationValidated) {
            if (websocket) {
                const token = localStorage.getItem('access_token');
                websocket.send(JSON.stringify({ 'check_in': token }));
                console.log('Sent check-in signal and access_token via WebSocket');
            }
        } else {
            // Show a notification for invalid conditions
            notification.error({
                message: 'Error',
                description: 'Invalid conditions for check-in. Please check face validation and location.',
                placement: 'topRight'
            });
        }
    };

    const turnOnCamera = () => {
        setWebcamVisible(true);
    };



    useEffect(() => {
        if (course) { // Kiểm tra xem course không phải là null hoặc undefined
            axiosInstance.get(`course/students/${course.course_id}/`)
                .then(response => {
                    setUsers(response.data);
                    response.data.forEach(user => {
                        axiosInstance.get(`/user/${user.staff_id}`)
                            .then(resp => {
                                setAvatars(prevAvatars => ({ ...prevAvatars, [user.staff_id]: resp.data.avatar }));
                                console.log(user.staff_id, resp.data.avatar);
                            })
                            .catch(error => console.error('Error fetching avatar:', error));
                    });
                })
                .catch(error => {
                    console.error('Lỗi khi tải dữ liệu users:', error);
                });

            const currentTime = new Date();
            const year = currentTime.getFullYear();
            const month = (currentTime.getMonth() + 1).toString().padStart(2, '0');
            const day = currentTime.getDate().toString().padStart(2, '0');
            const today = `${year}-${month}-${day}`;

            axiosInstance.get(`/attendance/?attendance_date=${today}&status=True&course_id=${course.course_id}&page_size=80`)
                .then(response => {
                    setPresentStudents(response.data.results.reduce((map, attendance) => {
                        map[attendance.student_id] = attendance.note;
                        return map;
                    }, {}));
                })
                .catch(error => {
                    console.error('Lỗi khi tải dữ liệu users:', error);
                });
        }
    }, [course]);

    return (
        <div style={modalStyles.AttendanceModal}>

            <div style={modalStyles.tableContainer}>
                <div style={modalStyles.tableWrapper}>
                    <table style={modalStyles.table}>
                        <thead>
                            <tr>
                                <th style={modalStyles.tableHeader}>Avatar</th>
                                <th style={modalStyles.tableHeader}>Mã số sinh viên</th>
                                <th style={modalStyles.tableHeader}>Họ và tên sinh viên</th>
                                <th style={modalStyles.tableHeader}>Môn học</th>
                                <th style={modalStyles.tableHeader}>Trạng thái điểm danh</th>
                                <th style={modalStyles.tableHeader}>Ghi chú</th>
                            </tr>
                        </thead>
                        <tbody>
                            {users && Array.isArray(users) ? (
                                users.map((user, index) => (
                                    <tr key={user.id}>
                                        <td style={modalStyles.cell}>
                                            {avatars[user.staff_id] &&
                                                <img
                                                    src={avatars[user.staff_id]}
                                                    style={{
                                                        width: '70px', // Tăng kích thước
                                                        height: '70px', // Tăng kích thước
                                                        borderRadius: '10px', // Bo tròn góc
                                                        objectFit: 'cover'
                                                    }} />}
                                        </td>
                                        <td style={modalStyles.cell}>{user.staff_id}</td>
                                        <td style={modalStyles.cell}>{user.full_name}</td>
                                        <td style={modalStyles.cell}>
                                            {course ? course.course_name : '-'}
                                        </td>
                                        <td style={modalStyles.cell}>
                                            {presentStudents[user.staff_id] ? (
                                                <span style={modalStyles.tick}>&#10003;</span>
                                            ) : (
                                                <span style={modalStyles.cross}>&#10007;</span>
                                            )}
                                            {presentStudents[user.staff_id] && presentStudents[user.staff_id].message}

                                        </td>
                                        <td style={modalStyles.cell}>{presentStudents[user.staff_id] || '-'}</td>
                                    </tr>
                                ))
                            ) : (
                                <tr>
                                    <td colSpan="6">No data available</td>
                                </tr>
                            )}
                        </tbody>

                    </table>
                    {isWithinTimeRange(course?.start_time, course?.end_time) ? (
                        <>
                            <>
                                <ButtonContainer>

                                    <button style={modalStyles.cameraButton} onClick={turnOnCamera} disabled={isFaceValidated}>
                                        {isFaceValidated ? <FaCheck /> : 'Bật camera điểm danh'}
                                    </button>
                                    <button style={modalStyles.checkLocationButton} onClick={handleCheckLocation} disabled={isLocationValidated}>
                                        {isLocationValidated ? <FaCheck /> : 'Kiểm tra vị trí'}
                                    </button>
                                </ButtonContainer>

                            </>
                            <div style={{ margin: '20px', textAlign: 'center' }}>
                                <button style={modalStyles.button} onClick={sendWebSocketData}>
                                    Điểm danh
                                </button>
                            </div>
                            {webcamVisible && (
                                <WebcamOverlay>
                                    <Webcam
                                        audio={false}
                                        videoConstraints={{ facingMode: 'user' }}
                                        ref={webcamRef}
                                    />
                                    <CaptureButton onClick={handleCapture}>Chụp ảnh</CaptureButton>
                                </WebcamOverlay>
                            )}
                        </>
                    ) : (
                        <button style={modalStyles.cameraButton} disabled>Bật camera điểm danh</button>
                    )}
                    {savedPosition && (
                        <div style={{ margin: '20px', textAlign: 'center' }}>
                            <h3>vị trí:</h3>
                            <p>Vĩ độ: {savedPosition.latitude}</p>
                            <p>Kinh độ: {savedPosition.longitude}</p>
                        </div>
                    )}
                    {capturedImage && (
                        <div>
                            <h3>Ảnh đã chụp:</h3>
                            <img src={capturedImage} alt="Captured" />
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default AttendanceModal;
