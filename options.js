function load() {
    let rateElement = document.getElementById('rate');
    let pitchElement = document.getElementById('pitch');
    let volumeElement = document.getElementById('volume');
    let rate = localStorage['rate'] || 1.0;
    let pitch = localStorage['pitch'] || 1.0;
    let volume = localStorage['volume'] || 1.0;
    rateElement.value = rate;
    pitchElement.value = pitch;
    volumeElement.value = volume;
    function listener(evt) {
        rate = rateElement.value;
        localStorage['rate'] = rate;
        pitch = pitchElement.value;
        localStorage['pitch'] = pitch;
        volume = volumeElement.value;
        localStorage['volume'] = volume;
    }
    rateElement.addEventListener('keyup', listener, false);
    pitchElement.addEventListener('keyup', listener, false);
    volumeElement.addEventListener('keyup', listener, false);
    rateElement.addEventListener('mouseup', listener, false);
    pitchElement.addEventListener('mouseup', listener, false);
    volumeElement.addEventListener('mouseup', listener, false);   

    let voice = document.getElementById('voice');
    let voiceArray = [];
    chrome.tts.getVoices(function(va) {
      voiceArray = va;
      for (let i = 0; i < voiceArray.length; i++) {
        let opt = document.createElement('option');
        let name = voiceArray[i].voiceName;
        if (name == localStorage['voice']) {
          opt.setAttribute('selected', '');
        }
        opt.setAttribute('value', name);
        opt.innerText = voiceArray[i].voiceName;
        voice.appendChild(opt);
      }
    });
    voice.addEventListener('change', function() {
      let i = voice.selectedIndex;
      localStorage['voice'] = voiceArray[i].voiceName;
    }, false);
}

document.addEventListener('DOMContentLoaded', load);