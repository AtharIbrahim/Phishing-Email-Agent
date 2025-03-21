function getEmailText() {
    let emailBody = document.querySelector(".a3s.aiL"); // Gmail email body class
    if (emailBody) {
        return emailBody.innerText;
    }
    return "No email content found.";
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.action === "getEmailContent") {
        let emailText = getEmailText();
        sendResponse({ emailText: emailText });
    }
});
