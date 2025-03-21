document.getElementById("checkEmail").addEventListener("click", () => {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        chrome.scripting.executeScript({
            target: { tabId: tabs[0].id },
            function: extractEmailContent
        }, (injectionResults) => {
            if (injectionResults && injectionResults[0].result) {
                let emailText = injectionResults[0].result;
                checkPhishing(emailText);
            } else {
                document.getElementById("result").innerText = "Error: Could not get email content.";
            }
        });
    });
});

function extractEmailContent() {
    let emailBody = document.querySelector(".a3s.aiL");
    return emailBody ? emailBody.innerText : "No email content found.";
}

function checkPhishing(emailText) {
    fetch("http://127.0.0.1:5000/detect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: emailText })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("result").innerText = "Result: " + data.result;
    })
    .catch(error => {
        document.getElementById("result").innerText = "Error connecting to backend.";
    });
}
