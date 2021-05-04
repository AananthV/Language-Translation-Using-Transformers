const englishTextarea = document.getElementById('english-textarea');
const tamilTextarea = document.getElementById('tamil-textarea');

const processing = false;

const translate = (text) => {
    const xhttp = new XMLHttpRequest();

    xhttp.onreadystatechange = function() {
        if (this.readyState == 4 && this.status == 200) {
          tamilTextarea.value = this.responseText;
        }
    };

    xhttp.open("POST", "/translate", true);
    xhttp.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
    xhttp.send("english=" + encodeURIComponent(text));
}

const onInputListener = (event) => {
    const englishText = englishTextarea.value;

    if (!processing) translate(englishText);
}

englishTextarea.addEventListener('input', onInputListener);