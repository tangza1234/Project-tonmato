// Tomato Leaf Disease Detection - frontend logic
// วางไฟล์นี้ที่ static/js/main.js

(function () {
    'use strict';

    var MAX_SIZE = 5 * 1024 * 1024; // 5MB
    var ALLOWED_TYPES = ['image/png', 'image/jpeg', 'image/webp'];

    var zone = document.getElementById('upload-zone');
    var input = document.getElementById('image');
    var fileError = document.getElementById('file-error');
    var fileNameDisplay = document.getElementById('file-name-display');
    var previewImg = document.getElementById('preview-img');
    var submitBtn = document.getElementById('submit-btn');
    var btnIcon = document.getElementById('btn-icon');
    var btnText = document.getElementById('btn-text');
    var serverError = document.getElementById('server-error');

    var resultCard = document.getElementById('result-card');
    var resultMain = document.getElementById('result-main');
    var resultIcon = document.getElementById('result-icon');
    var resultName = document.getElementById('result-name');
    var confPct = document.getElementById('conf-pct');
    var confBar = document.getElementById('conf-bar');
    var confLabel = confPct.parentElement;   // บรรทัด "ความมั่นใจ: xx%"
    var confWrap = confBar.parentElement;    // แถบความมั่นใจ
    var btnCopy = document.getElementById('btn-copy');

    var diseaseSection = document.getElementById('disease-info-section');
    var infoCause = document.getElementById('info-cause');
    var infoSymptom = document.getElementById('info-symptom');
    var infoTreatment = document.getElementById('info-treatment');

    var top3Section = document.getElementById('top3-section');
    var top3List = document.getElementById('top3-list');

    var imgSection = document.getElementById('img-section');
    var resultImg = document.getElementById('result-img');

    var selectedFile = null;
    var previewUrl = null;
    var lastCopyText = '';

    var ICONS = {
        disease: '🦠',
        healthy: '✅',
        unknown: '❓',
        not_tomato: '🚫'
    };

    // ---------- helpers ----------

    function showFileError(msg) {
        fileError.textContent = msg;
        fileError.style.display = 'block';
        fileError.style.color = '#c1121f';
        fileError.style.fontSize = '0.85rem';
        fileError.style.fontWeight = '600';
        fileError.style.marginTop = '10px';
    }

    function clearFileError() {
        fileError.textContent = '';
        fileError.style.display = 'none';
    }

    function showServerError(msg) {
        serverError.textContent = '⚠️ ' + msg;
        serverError.style.display = 'flex';
    }

    function clearServerError() {
        serverError.textContent = '';
        serverError.style.display = 'none';
    }

    function setLoading(isLoading) {
        if (isLoading) {
            submitBtn.classList.add('loading');
            submitBtn.disabled = true;
            btnIcon.textContent = '⏳';
            btnText.textContent = 'กำลังวิเคราะห์...';
        } else {
            submitBtn.classList.remove('loading');
            submitBtn.disabled = !selectedFile;
            btnIcon.textContent = '🔍';
            btnText.textContent = 'วินิจฉัยโรค';
        }
    }

    function resetSelection() {
        selectedFile = null;
        if (previewUrl) {
            URL.revokeObjectURL(previewUrl);
            previewUrl = null;
        }
        previewImg.removeAttribute('src');
        previewImg.style.display = 'none';
        fileNameDisplay.textContent = '';
        submitBtn.disabled = true;
    }

    function formatSize(bytes) {
        if (bytes < 1024 * 1024) {
            return (bytes / 1024).toFixed(0) + ' KB';
        }
        return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
    }

    // ---------- file selection ----------

    function handleFile(file) {
        clearFileError();
        clearServerError();
        resultCard.style.display = 'none';

        if (!file) {
            resetSelection();
            return;
        }

        if (ALLOWED_TYPES.indexOf(file.type) === -1) {
            resetSelection();
            input.value = '';
            showFileError('รองรับเฉพาะไฟล์ PNG, JPG, JPEG, WEBP เท่านั้น');
            return;
        }

        if (file.size > MAX_SIZE) {
            resetSelection();
            input.value = '';
            showFileError('ไฟล์ใหญ่เกิน 5MB (' + formatSize(file.size) + ') กรุณาเลือกไฟล์ที่เล็กกว่านี้');
            return;
        }

        selectedFile = file;

        if (previewUrl) {
            URL.revokeObjectURL(previewUrl);
        }
        previewUrl = URL.createObjectURL(file);
        previewImg.src = previewUrl;
        previewImg.style.display = 'block';

        fileNameDisplay.textContent = file.name + ' · ' + formatSize(file.size);
        submitBtn.disabled = false;
    }

    input.addEventListener('change', function () {
        handleFile(input.files && input.files[0]);
    });

    // drag & drop
    ['dragenter', 'dragover'].forEach(function (evt) {
        zone.addEventListener(evt, function (e) {
            e.preventDefault();
            zone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(function (evt) {
        zone.addEventListener(evt, function (e) {
            e.preventDefault();
            zone.classList.remove('dragover');
        });
    });

    zone.addEventListener('drop', function (e) {
        var files = e.dataTransfer && e.dataTransfer.files;
        if (files && files.length) {
            try {
                input.files = files;
            } catch (err) { /* บางเบราว์เซอร์ไม่อนุญาต ไม่เป็นไร */ }
            handleFile(files[0]);
        }
    });

    // วางรูปด้วย Ctrl+V
    document.addEventListener('paste', function (e) {
        var items = e.clipboardData && e.clipboardData.items;
        if (!items) return;
        for (var i = 0; i < items.length; i++) {
            if (items[i].kind === 'file' && items[i].type.indexOf('image/') === 0) {
                var file = items[i].getAsFile();
                if (file) {
                    handleFile(file);
                    break;
                }
            }
        }
    });

    // ---------- submit ----------

    submitBtn.addEventListener('click', function () {
        if (!selectedFile) return;

        clearServerError();
        resultCard.style.display = 'none';
        setLoading(true);

        var formData = new FormData();
        formData.append('image', selectedFile);

        fetch('/predict', { method: 'POST', body: formData })
            .then(function (res) {
                return res.text().then(function (text) {
                    var data = null;
                    try {
                        data = JSON.parse(text);
                    } catch (err) {
                        data = null;
                    }
                    if (!res.ok) {
                        var msg = (data && data.error) ||
                            'เซิร์ฟเวอร์ผิดพลาด (รหัส ' + res.status + ') กรุณาลองใหม่อีกครั้ง';
                        throw new Error(msg);
                    }
                    if (!data) {
                        throw new Error('ได้รับข้อมูลจากเซิร์ฟเวอร์ในรูปแบบที่ไม่ถูกต้อง');
                    }
                    return data;
                });
            })
            .then(function (data) {
                renderResult(data);
            })
            .catch(function (err) {
                if (err instanceof TypeError) {
                    showServerError('เชื่อมต่อเซิร์ฟเวอร์ไม่ได้ กรุณาตรวจสอบอินเทอร์เน็ตแล้วลองใหม่');
                } else {
                    showServerError(err.message);
                }
            })
            .then(function () {
                setLoading(false);
            });
    });

    // ---------- render result ----------

    function renderResult(data) {
        var type = data.result_type || 'unknown';
        var confidence = Number(data.confidence) || 0;

        resultMain.className = 'result-main ' + type;
        resultIcon.textContent = ICONS[type] || '❓';
        resultName.textContent = data.result || '-';
        confPct.textContent = confidence.toFixed(2) + '%';

        // กรณีเซิร์ฟเวอร์ไม่ฟันธง (hide_confidence) ซ่อนตัวเลขและแถบความมั่นใจ
        var hideConf = !!data.hide_confidence;
        confLabel.style.display = hideConf ? 'none' : '';
        confWrap.style.display = hideConf ? 'none' : '';

        // เริ่มจาก 0 แล้วค่อยขยาย เพื่อให้มีแอนิเมชัน
        confBar.style.width = '0%';
        requestAnimationFrame(function () {
            requestAnimationFrame(function () {
                confBar.style.width = Math.min(confidence, 100) + '%';
            });
        });

        // ข้อมูลโรค
        var info = data.disease_info;
        if (info) {
            infoCause.textContent = info.cause || '-';
            infoSymptom.textContent = info.symptom || '-';
            infoTreatment.textContent = info.treatment || '-';
            diseaseSection.style.display = 'block';
        } else {
            diseaseSection.style.display = 'none';
        }

        // Top 3
        top3List.innerHTML = '';
        if (data.top3 && data.top3.length) {
            data.top3.forEach(function (item, i) {
                var row = document.createElement('div');
                row.className = 'top3-item';

                var rank = document.createElement('div');
                rank.className = 'top3-rank';
                rank.textContent = '#' + (i + 1);

                var name = document.createElement('div');
                name.className = 'top3-name';
                name.textContent = item.name || item['class'];
                name.title = item.name || item['class'];

                var barWrap = document.createElement('div');
                barWrap.className = 'top3-bar-wrap';
                var barFill = document.createElement('div');
                barFill.className = 'top3-bar-fill';
                barFill.style.width = Math.min(Number(item.confidence) || 0, 100) + '%';
                barWrap.appendChild(barFill);

                var pct = document.createElement('div');
                pct.className = 'top3-pct';
                pct.textContent = (Number(item.confidence) || 0).toFixed(1) + '%';

                row.appendChild(rank);
                row.appendChild(name);
                row.appendChild(barWrap);
                row.appendChild(pct);
                top3List.appendChild(row);
            });
            top3Section.style.display = 'block';
        } else {
            top3Section.style.display = 'none';
        }

        // รูปที่อัปโหลด (ใช้ preview ในเครื่อง ไม่ต้องรอโหลดจากเซิร์ฟเวอร์)
        if (previewUrl) {
            resultImg.src = previewUrl;
            imgSection.style.display = 'block';
        } else if (data.image_path) {
            resultImg.src = data.image_path;
            imgSection.style.display = 'block';
        } else {
            imgSection.style.display = 'none';
        }

        // ข้อความสำหรับปุ่มคัดลอก
        var lines = ['ผลการวินิจฉัย: ' + (data.result || '-')];
        if (!hideConf) {
            lines.push('ความมั่นใจ: ' + confidence.toFixed(2) + '%');
        }
        if (info) {
            lines.push('สาเหตุ: ' + (info.cause || '-'));
            lines.push('อาการ: ' + (info.symptom || '-'));
            lines.push('การรักษา: ' + (info.treatment || '-'));
        }
        lastCopyText = lines.join('\n');

        resultCard.style.display = 'block';
        resultCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    // ---------- copy ----------

    btnCopy.addEventListener('click', function () {
        if (!lastCopyText) return;

        function done() {
            var old = btnCopy.textContent;
            btnCopy.textContent = '✔️';
            setTimeout(function () { btnCopy.textContent = old; }, 1500);
        }

        if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(lastCopyText).then(done).catch(function () {
                fallbackCopy(lastCopyText);
                done();
            });
        } else {
            fallbackCopy(lastCopyText);
            done();
        }
    });

    function fallbackCopy(text) {
        var ta = document.createElement('textarea');
        ta.value = text;
        ta.style.position = 'fixed';
        ta.style.opacity = '0';
        document.body.appendChild(ta);
        ta.select();
        try { document.execCommand('copy'); } catch (err) { /* ignore */ }
        document.body.removeChild(ta);
    }
})();
