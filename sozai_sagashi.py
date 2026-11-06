"""素材さがし (Sozai Sagashi) — CLIP STUDIO PAINT の素材を自然な言葉で検索するツール.

- サムネイル検索: SigLIP2 (画像 ↔ 多言語テキスト)
- テキスト検索 : BGE-M3 (素材名・タグ・説明・パレットフォルダ ↔ 多言語テキスト)
- CLIP STUDIO の素材DBはコピーを読み取るだけ (元データは変更しません)

MIT License (c) 2026 Behelit / Developed with Claude AI, Codex
"""
import json, os, pathlib, pickle, re, sqlite3, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
from PySide6 import QtCore, QtGui, QtWidgets

VERSION = '1.0.0'
APP_ID = 'Behelit.SozaiSagashi'
HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(HERE, 'config.json')
CACHE = os.path.join(HERE, 'index_cache.pkl')
IMG_MODEL = 'google/siglip2-so400m-patch16-256'
TXT_MODEL = 'BAAI/bge-m3'
MAX_RESULTS = 150
BATCH = 32  # 解析のバッチ. 128 だと VRAM 4GiB 制限で不足 (実測: 32 で最大 3.4GiB)

# ---------------------------------------------------------------- 言語 / 언어

STR = {
    'ja': {
        'title': '素材さがし',
        'placeholder': '探している素材を言葉で入力（例：雨の夜の路地裏 / 桜吹雪のブラシ / 足を組んで座るポーズ）※日本語・韓国語・英語OK',
        'search': '検索', 'refresh': '再読み込み', 'about': '情報', 'folder': 'フォルダ…',
        'refresh_tip': 'CLIP STUDIO で素材を新しくダウンロードしたときに押してください',
        'folder_tip': 'CLIP STUDIO の素材の保存場所（CLIPStudioCommon フォルダ）を指定します',
        'pick': '素材の保存場所（CLIPStudioCommon フォルダ）を選んでください',
        'pick_bad': 'このフォルダには CLIP STUDIO の素材データ（MaterialDB）がありません。',
        'tab_results': '検索結果', 'tab_favs': '★ お気に入り ({n})',
        'fav_add': '☆ お気に入り', 'fav_del': '★ お気に入り解除', 'copy': '名前をコピー',
        'help': 'ダブルクリックで素材名をコピー → CLIP STUDIO の素材パレットの検索欄に貼り付けてください。',
        'where': 'パレットの場所: {w}',
        'copied': 'コピーしました：「{n}」 — 素材パレットの検索欄に貼り付けてください',
        'loading': 'AIモデルを読み込み中…',
        'reading': '素材リストを読み込み中…',
        'idx_img': 'サムネイルを解析中 {i}/{n}（初回のみ数分かかります）',
        'idx_txt': '素材名・タグを解析中 {i}/{n}',
        'ready': '準備完了 — 素材 {n:,} 件',
        'no_csp': 'CLIP STUDIO の素材データが見つかりません — 「フォルダ…」から保存場所を指定してください',
        'error': 'エラー: {e}',
        'filters': ['すべて', '画像', 'ブラシ・ツール', 'ポーズ', '3D', 'テンプレート・フキダシ'],
        'about_text': ('<b>素材さがし</b> v{v}<br><br>'
                       'CLIP STUDIO PAINT の素材を、日本語・韓国語・英語など<br>好きな言葉で探せるツールです。<br><br>'
                       '© 2026 Behelit — MIT License<br>Developed with Claude AI, Codex<br><br>'
                       '<small>非公式ツールです。CLIP STUDIO PAINT は株式会社セルシスの商標です。<br>'
                       'AIモデル: SigLIP 2 (Apache-2.0), BGE-M3 (MIT)</small>'),
    },
    'ko': {
        'title': '소재 찾기',
        'placeholder': '찾는 소재를 말로 설명하세요 (예: 비 오는 밤 골목 / 벚꽃 흩날리는 브러시 / 다리 꼬고 앉은 포즈) ※한국어·일본어·영어 OK',
        'search': '검색', 'refresh': '새로고침', 'about': '정보', 'folder': '폴더…',
        'refresh_tip': '클립스튜디오에서 소재를 새로 다운로드했을 때 눌러 주세요',
        'folder_tip': '클립스튜디오 소재 저장 위치(CLIPStudioCommon 폴더)를 지정합니다',
        'pick': '소재 저장 위치(CLIPStudioCommon 폴더)를 선택하세요',
        'pick_bad': '이 폴더에는 클립스튜디오 소재 데이터(MaterialDB)가 없습니다.',
        'tab_results': '검색 결과', 'tab_favs': '★ 즐겨찾기 ({n})',
        'fav_add': '☆ 즐겨찾기', 'fav_del': '★ 즐겨찾기 해제', 'copy': '이름 복사',
        'help': '더블클릭하면 소재 이름이 복사됩니다 → 클립스튜디오 소재 팔레트 검색창에 붙여넣으세요.',
        'where': '팔레트 위치: {w}',
        'copied': '복사했습니다: 「{n}」 — 소재 팔레트 검색창에 붙여넣으세요',
        'loading': 'AI 모델 불러오는 중…',
        'reading': '소재 목록 읽는 중…',
        'idx_img': '썸네일 분석 중 {i}/{n} (처음 한 번만 몇 분 걸려요)',
        'idx_txt': '소재 이름·태그 분석 중 {i}/{n}',
        'ready': '준비 완료 — 소재 {n:,}개',
        'no_csp': '클립스튜디오 소재 데이터를 찾을 수 없습니다 — "폴더…"에서 저장 위치를 지정하세요',
        'error': '오류: {e}',
        'filters': ['전체', '이미지', '브러시·도구', '포즈', '3D', '템플릿·말풍선'],
        'about_text': ('<b>소재 찾기 (素材さがし)</b> v{v}<br><br>'
                       '클립스튜디오 소재를 한국어·일본어·영어 등<br>편한 말로 찾을 수 있는 도구입니다.<br><br>'
                       '© 2026 Behelit — MIT License<br>Developed with Claude AI, Codex<br><br>'
                       '<small>비공식 도구입니다. CLIP STUDIO PAINT는 CELSYS의 상표입니다.<br>'
                       'AI 모델: SigLIP 2 (Apache-2.0), BGE-M3 (MIT)</small>'),
    },
}
KIND = {  # kind: (ja, ko)
    'image': ('画像', '이미지'), 'brush': ('ブラシ', '브러시'), 'pose': ('ポーズ', '포즈'),
    '3dhand': ('3D手', '3D 손'), 'other_tool': ('ツール', '도구'), 'gradation': ('グラデーション', '그라데이션'),
    'gradation_set': ('グラデーションセット', '그라데이션 세트'), 'color_set': ('カラーセット', '컬러 세트'),
    'auto_action': ('オートアクション', '오토 액션'), 'smallobject_v4': ('3D小物', '3D 소품'),
    'smallobject': ('3D小物', '3D 소품'), 'room_v4': ('3D背景', '3D 배경'), 'character_v4': ('3Dキャラクター', '3D 캐릭터'),
    '3dprimitive': ('3Dプリミティブ', '3D 도형'), '3dhead': ('3D頭部', '3D 머리'), 'proportion': ('3D体型', '3D 체형'),
    'skysphere': ('スカイスフィア', '스카이 스피어'), 'layer_template': ('レイヤーテンプレート', '레이어 템플릿'),
    'frame_template': ('コマ枠テンプレート', '컷 템플릿'), 'balloon': ('フキダシ', '말풍선'),
}
FILTER_KINDS = [
    None,
    {'image'},
    {'brush', 'other_tool', 'gradation', 'gradation_set', 'color_set', 'auto_action'},
    {'pose', '3dhand'},
    {'smallobject_v4', 'smallobject', 'room_v4', 'character_v4', '3dprimitive', '3dhead', 'proportion', 'skysphere'},
    {'layer_template', 'frame_template', 'balloon'},
]
LANG = 'ja'


def T(key, **kw):
    s = STR[LANG][key]
    return s.format(**kw) if kw else s


def kind_name(kind):
    return KIND.get(kind, (kind, kind))[LANG == 'ko']


def load_config():
    try:
        with open(CONFIG, encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {'lang': 'ja', 'favorites': []}


def save_config(cfg):
    with open(CONFIG, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def load_img(path, size=256):
    """サムネイルを白背景の正方形RGBに. 失敗時 None."""
    try:
        im = Image.open(path)
        im.thumbnail((size, size))
        im = im.convert('RGBA')
        bg = Image.new('RGB', (size, size), 'white')
        bg.paste(im, ((size - im.width) // 2, (size - im.height) // 2), im)
        return bg
    except Exception:
        return None


# ---------------------------------------------------------------- 素材リスト

def db_file(common):
    return os.path.join(common, 'MaterialDB', 'CatalogMaterial.cmdb')


def find_csp_dir(chosen):
    """素材の保存場所 (CLIPStudioCommon). 指定があればそれ, 無ければ既定の2か所のうちDBが新しい方."""
    if chosen and os.path.exists(db_file(chosen)):
        return chosen
    docs = os.path.normpath(QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.StandardLocation.DocumentsLocation))
    cands = [os.path.expandvars(r'%APPDATA%\CELSYSUserData\CELSYS\CLIPStudioCommon'),  # Ver.1.10.13 以降
             os.path.join(docs, 'CELSYS', 'CLIPStudioCommon')]                           # Ver.1.10.12 以前
    found = [d for d in cands if os.path.exists(db_file(d))]
    return max(found, key=lambda d: os.path.getmtime(db_file(d))) if found else None


def load_csp_items(common):
    tmp = os.path.join(HERE, '_csp_db_copy.cmdb')
    if os.path.exists(tmp):
        os.remove(tmp)
    # 使用中のDBを単純コピーすると壊れた複製になり得る → 読み取り専用で開き SQLite の backup で複製
    src = sqlite3.connect(pathlib.Path(db_file(common)).as_uri() + '?mode=ro', uri=True)
    c = sqlite3.connect(tmp)
    src.backup(c)
    src.close()
    c.execute('create index if not exists _tag_el on TableTagElementMap(ElementId)')  # コピーにだけ. 無いと数分かかる
    c.execute('create index if not exists _folder_el on TableFolderTagElementMap(ElementId)')
    rows = c.execute('''
        select m.MaterialName, m.MaterialUuid, m.Path, d.MaterialKind, d.Lead, f.Title,
          (select group_concat(t.Title, ' ') from TableTagElementMap x join Tag t on t._PW_ID = x.TagId
           where x.ElementId = m._PW_ID)
        from MaterialModifier m join CatalogMetaData d on d.ModifierID = m._PW_ID
        left join TableFolderTagElementMap fm on fm.ElementId = m._PW_ID
        left join FolderTag f on f._PW_ID = fm.TagId''').fetchall()
    c.close()
    # セット(catalog)の名前・タグ・説明は、中の個別素材のテキストに付ける
    sets = {path: ' '.join(filter(None, (name, tags, lead)))
            for name, uuid, path, kind, lead, folder, tags in rows if kind == 'catalog'}
    items = []
    for name, uuid, path, kind, lead, folder, tags in rows:
        if kind == 'catalog' or not path or not uuid:
            continue
        d = os.path.join(common, 'Material', *path.split(':')[1:])
        thumb = next((f for f in (os.path.join(d, uuid, 'thumbnail', 'thumbnail.png'),
                                  os.path.join(d, 'thumbnail', 'thumbnail.png')) if os.path.exists(f)), None)
        where = ' > '.join(re.sub(r'-\d{5,}$', '', s) for s in (folder or '').split(':') if s)
        text = ' '.join(filter(None, (name, tags, lead, where, sets.get(path))))[:600]
        items.append({'id': f'csp:{path}:{uuid}', 'name': name or '-', 'kind': kind,
                      'thumb': thumb, 'text': text, 'where': where})
    return items


# ---------------------------------------------------------------- 埋め込み

class Engine:
    def __init__(self):
        import torch
        from transformers import AutoModel, AutoProcessor, AutoTokenizer
        self.torch = torch
        # このPyTorchが対応していない古いGPU (GTX 10シリーズ以前など) は CPU で動かす
        archs = ([int(a[3:]) for a in torch.cuda.get_arch_list() if a.startswith('sm_')]
                 if torch.cuda.is_available() and torch.cuda.device_count() else [])
        cap = torch.cuda.get_device_capability() if archs else (0, 0)
        self.dev = 'cuda' if archs and cap[0] * 10 + cap[1] >= min(archs) else 'cpu'
        dt = torch.float16 if self.dev == 'cuda' else torch.float32
        self.im = self._load(AutoModel, IMG_MODEL, dtype=dt).to(self.dev).eval()
        self.ip = self._load(AutoProcessor, IMG_MODEL)
        self.tm = self._load(AutoModel, TXT_MODEL, dtype=dt).to(self.dev).eval()
        self.tt = self._load(AutoTokenizer, TXT_MODEL)

    @staticmethod
    def _load(cls, name, **kw):
        """setup.bat で取得済みのモデルを使う (起動のたびにネットへ確認しない → 通信エラーで起動失敗しない). 無いときだけDL."""
        try:
            return cls.from_pretrained(name, local_files_only=True, **kw)
        except OSError:
            return cls.from_pretrained(name, **kw)

    @staticmethod
    def _vec(out):
        if not hasattr(out, 'norm'):  # transformers のバージョンによって ModelOutput で返る
            out = out.pooler_output
        return (out / out.norm(dim=-1, keepdim=True)).float().cpu().numpy()

    def images(self, pil_list):
        with self.torch.inference_mode():
            inp = self.ip(images=pil_list, return_tensors='pt').to(self.dev)
            inp['pixel_values'] = inp['pixel_values'].to(self.im.dtype)
            return self._vec(self.im.get_image_features(**inp))

    def query_for_images(self, q):
        with self.torch.inference_mode():
            inp = self.ip(text=[q.lower()], padding='max_length', max_length=64,
                          truncation=True, return_tensors='pt').to(self.dev)
            return self._vec(self.im.get_text_features(**inp))[0]

    def texts(self, texts):
        with self.torch.inference_mode():
            inp = self.tt(texts, padding=True, truncation=True, max_length=256, return_tensors='pt').to(self.dev)
            return self._vec(self.tm(**inp).last_hidden_state[:, 0])


class Indexer(QtCore.QThread):
    progress = QtCore.Signal(str)
    finished_ok = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, engine, common):
        super().__init__()
        self.engine, self.common = engine, common
        self.stop = False  # 解析中にウィンドウを閉じたら True → 区切りで止める

    def run(self):
        try:
            self._run()
        except Exception as e:  # 失敗しても「再読み込み」でやり直せるように画面へ知らせる
            self.failed.emit(repr(e))

    def _run(self):
        if self.engine is None:
            self.progress.emit(T('loading'))
            self.engine = Engine()
        self.progress.emit(T('reading'))
        items = load_csp_items(self.common)
        try:
            with open(CACHE, 'rb') as f:
                cache = pickle.load(f)
        except Exception:
            cache = {'img': {}, 'txt': {}}

        img_keys = []
        for it in items:
            k = None
            if it['thumb']:
                try:
                    k = f"{it['thumb']}|{os.path.getmtime(it['thumb'])}"
                except OSError:
                    pass
            img_keys.append(k)

        todo = [k for k in dict.fromkeys(img_keys) if k and k not in cache['img']]
        with ThreadPoolExecutor(8) as pool:
            for i in range(0, len(todo), BATCH):
                if self.stop:
                    break
                self.progress.emit(T('idx_img', i=i, n=len(todo)))
                batch = todo[i:i + BATCH]
                pils = list(pool.map(lambda k: load_img(k.rsplit('|', 1)[0]), batch))
                ok = [(k, p) for k, p in zip(batch, pils) if p is not None]
                if ok:
                    for (k, _), v in zip(ok, self.engine.images([p for _, p in ok])):
                        cache['img'][k] = v.astype(np.float16)
                for k, p in zip(batch, pils):
                    if p is None:
                        cache['img'][k] = None

        todo = [t for t in dict.fromkeys(it['text'] for it in items) if t and t not in cache['txt']]
        for i in range(0, len(todo), BATCH):
            if self.stop:
                break
            self.progress.emit(T('idx_txt', i=i, n=len(todo)))
            for t, v in zip(todo[i:i + BATCH], self.engine.texts(todo[i:i + BATCH])):
                cache['txt'][t] = v.astype(np.float16)
        self.engine.torch.cuda.empty_cache()  # 解析用の一時メモリをGPUに返す (常駐はモデル分だけ)

        # 今使っているものだけ残して保存 (途中で止めた場合も、そこまでの結果は次回に使う)
        used_img, used_txt = set(img_keys), {it['text'] for it in items}
        cache = {'img': {k: v for k, v in cache['img'].items() if k in used_img},
                 'txt': {k: v for k, v in cache['txt'].items() if k in used_txt}}
        with open(CACHE, 'wb') as f:
            pickle.dump(cache, f)
        if self.stop:
            return

        di = next((len(v) for v in cache['img'].values() if v is not None), 1152)
        dt = next(iter(cache['txt'].values()), np.zeros(1024)).shape[0]
        IMG = np.full((len(items), di), np.nan, np.float32)
        TXT = np.full((len(items), dt), np.nan, np.float32)
        for n, (it, k) in enumerate(zip(items, img_keys)):
            if k and cache['img'].get(k) is not None:
                IMG[n] = cache['img'][k]
            if it['text']:
                TXT[n] = cache['txt'][it['text']]
        self.finished_ok.emit((self.engine, items, IMG, TXT))


# ---------------------------------------------------------------- UI

class ResultList(QtWidgets.QListWidget):
    def __init__(self):
        super().__init__()
        self.setViewMode(QtWidgets.QListView.IconMode)
        self.setIconSize(QtCore.QSize(128, 128))
        self.setGridSize(QtCore.QSize(150, 180))
        self.setResizeMode(QtWidgets.QListView.Adjust)
        self.setMovement(QtWidgets.QListView.Static)
        self.setWordWrap(True)
        self.setUniformItemSizes(True)
        self._queue = []

    def show_items(self, its):
        self.clear()
        blank = QtGui.QPixmap(128, 128)
        blank.fill(QtGui.QColor('#ddd'))
        self._queue = []
        for it in its:
            w = QtWidgets.QListWidgetItem(QtGui.QIcon(blank), it['name'])
            w.setData(QtCore.Qt.UserRole, it)
            w.setToolTip(f"{it['name']}\n[{kind_name(it['kind'])}] {it['where']}")
            self.addItem(w)
            self._queue.append(w)
        QtCore.QTimer.singleShot(0, self._load_icons)

    def _load_icons(self):
        """サムネイルは少しずつ読み込む (UIを止めない)."""
        for w in self._queue[:16]:
            it = w.data(QtCore.Qt.UserRole)
            im = load_img(it['thumb'], 128) if it['thumb'] else None
            if im:
                qi = QtGui.QImage(im.tobytes(), im.width, im.height, 3 * im.width, QtGui.QImage.Format_RGB888).copy()
                w.setIcon(QtGui.QIcon(QtGui.QPixmap.fromImage(qi)))
        self._queue = self._queue[16:]
        if self._queue:
            QtCore.QTimer.singleShot(0, self._load_icons)


class Main(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        global LANG
        self.cfg = load_config()
        LANG = self.cfg.get('lang', 'ja')
        self.resize(980, 760)
        self.engine, self.items, self.IMG, self.TXT, self.low = None, [], None, None, []
        self.indexer, self.status = None, ('loading', {})

        self.q = QtWidgets.QLineEdit()
        self.q.returnPressed.connect(self.search)
        self.flt = QtWidgets.QComboBox()
        self.flt.currentIndexChanged.connect(lambda: self.q.text().strip() and self.search())
        self.btn_go = QtWidgets.QPushButton()
        self.btn_go.clicked.connect(self.search)
        self.btn_refresh = QtWidgets.QPushButton()
        self.btn_refresh.clicked.connect(self.reindex)
        self.btn_folder = QtWidgets.QPushButton()
        self.btn_folder.clicked.connect(lambda: self.pick_folder() and self.reindex())
        self.lang = QtWidgets.QComboBox()
        self.lang.addItems(['日本語', '한국어'])
        self.lang.setCurrentIndex(int(LANG == 'ko'))
        self.lang.currentIndexChanged.connect(self.set_lang)
        self.btn_about = QtWidgets.QPushButton()
        self.btn_about.clicked.connect(lambda: QtWidgets.QMessageBox.about(self, T('about'), T('about_text', v=VERSION)))
        top = QtWidgets.QHBoxLayout()
        top.addWidget(self.q, 1)
        for wd in (self.flt, self.btn_go, self.btn_refresh, self.btn_folder, self.lang, self.btn_about):
            top.addWidget(wd)

        self.results, self.favs = ResultList(), ResultList()
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self.results, '')
        self.tabs.addTab(self.favs, '')
        self.tabs.currentChanged.connect(self.on_select)
        for lst in (self.results, self.favs):
            lst.currentItemChanged.connect(self.on_select)
            lst.itemDoubleClicked.connect(lambda w: self.copy(w.data(QtCore.Qt.UserRole)))

        self.info = QtWidgets.QLabel('')
        self.info.setWordWrap(True)
        self.info.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.btn_fav = QtWidgets.QPushButton()
        self.btn_copy = QtWidgets.QPushButton()
        self.btn_fav.clicked.connect(self.toggle_fav)
        self.btn_copy.clicked.connect(lambda: self.copy(self.current()))
        bottom = QtWidgets.QHBoxLayout()
        bottom.addWidget(self.info, 1)
        bottom.addWidget(self.btn_fav)
        bottom.addWidget(self.btn_copy)

        lay = QtWidgets.QVBoxLayout()
        lay.addLayout(top)
        lay.addWidget(self.tabs, 1)
        lay.addLayout(bottom)
        w = QtWidgets.QWidget()
        w.setLayout(lay)
        self.setCentralWidget(w)
        self.retranslate()
        self.set_busy(True)
        QtCore.QTimer.singleShot(0, self.reindex)  # 先にウィンドウを出してから (フォルダ選択が出ることがある)

    # --- 言語
    def set_lang(self, idx):
        global LANG
        LANG = 'ko' if idx else 'ja'
        self.cfg['lang'] = LANG
        save_config(self.cfg)
        self.retranslate()

    def retranslate(self):
        self.setWindowTitle(T('title'))
        self.q.setPlaceholderText(T('placeholder'))
        self.btn_go.setText(T('search'))
        self.btn_refresh.setText(T('refresh'))
        self.btn_refresh.setToolTip(T('refresh_tip'))
        self.btn_folder.setText(T('folder'))
        self.btn_folder.setToolTip(T('folder_tip'))
        self.btn_about.setText(T('about'))
        self.btn_copy.setText(T('copy'))
        i = max(self.flt.currentIndex(), 0)
        self.flt.blockSignals(True)
        self.flt.clear()
        self.flt.addItems(T('filters'))
        self.flt.setCurrentIndex(i)
        self.flt.blockSignals(False)
        self.tabs.setTabText(0, T('tab_results'))
        self.tabs.setTabText(1, T('tab_favs', n=self.favs.count()))
        self.show_status(*self.status)
        self.on_select()

    def show_status(self, key, kw):
        self.status = (key, kw)
        self.statusBar().showMessage(T(key, **kw))

    # --- インデックス
    def set_busy(self, busy):
        for wd in (self.q, self.btn_go):
            wd.setEnabled(not busy and self.IMG is not None)  # 検索は索引があるときだけ
        for wd in (self.btn_refresh, self.btn_folder):
            wd.setEnabled(not busy)

    def reindex(self):
        common = find_csp_dir(self.cfg.get('csp_dir'))
        if not common and self.pick_folder():
            common = self.cfg['csp_dir']
        if not common:
            self.show_status('no_csp', {})
            self.set_busy(False)
            return
        self.set_busy(True)
        self.indexer = Indexer(self.engine, common)
        self.indexer.progress.connect(self.statusBar().showMessage)
        self.indexer.finished_ok.connect(self.on_indexed)
        self.indexer.failed.connect(self.on_failed)
        self.indexer.start()

    def pick_folder(self):
        """素材の保存場所 (CLIPStudioCommon フォルダ) を選んで保存. 正しいフォルダなら True."""
        d = QtWidgets.QFileDialog.getExistingDirectory(self, T('pick'))
        for c in ((d, os.path.join(d, 'CLIPStudioCommon'), os.path.join(d, 'CELSYS', 'CLIPStudioCommon')) if d else ()):
            if os.path.exists(db_file(c)):
                self.cfg['csp_dir'] = os.path.normpath(c)
                save_config(self.cfg)
                return True
        if d:
            QtWidgets.QMessageBox.warning(self, T('title'), T('pick_bad'))
        return False

    def on_indexed(self, res):
        self.engine, self.items, self.IMG, self.TXT = res
        self.low = [it['text'].lower() + ' ' + it['name'].lower() for it in self.items]
        self.show_status('ready', {'n': len(self.items)})
        self.set_busy(False)
        self.q.setFocus()
        self.results.clear()  # 再読み込み前の古い結果 (削除・改名された素材) を残さない
        self.search()  # 検索語があれば新しい索引で検索し直す
        self.refresh_favs()

    def on_failed(self, msg):
        self.engine = self.indexer.engine  # モデルの読み込みまで済んでいれば再利用
        self.show_status('error', {'e': msg})
        self.set_busy(False)  # 「再読み込み」でやり直せる (前の索引があれば検索もできる)

    def closeEvent(self, e):
        if self.indexer and self.indexer.isRunning():  # 解析中なら区切りで止めてから終了 (途中までの結果は保存)
            self.indexer.stop = True
            self.hide()
            self.indexer.wait()
        e.accept()

    # --- 検索
    def search(self):
        q = self.q.text().strip()
        if not q or self.IMG is None or not self.items:
            return
        z = lambda s: np.nan_to_num((s - np.nanmean(s)) / (np.nanstd(s) + 1e-6))
        score = 0.6 * z(self.IMG @ self.engine.query_for_images(q)) + 0.4 * z(self.TXT @ self.engine.texts([q])[0])
        # 入力した文がそのまま素材名・タグに含まれていれば加点
        score += np.fromiter((q.lower() in t for t in self.low), bool, len(self.low))
        kinds = FILTER_KINDS[self.flt.currentIndex()]
        if kinds:
            score[[it['kind'] not in kinds for it in self.items]] = -np.inf
        top = [i for i in np.argsort(-score)[:MAX_RESULTS] if np.isfinite(score[i])]
        self.results.show_items([self.items[i] for i in top])
        self.tabs.setCurrentWidget(self.results)

    # --- 選択/操作
    def current(self):
        w = self.tabs.currentWidget().currentItem()
        return w.data(QtCore.Qt.UserRole) if w else None

    def on_select(self, *_):
        it = self.current()
        for b in (self.btn_fav, self.btn_copy):
            b.setEnabled(it is not None)
        if not it:
            self.info.setText(T('help'))
            self.btn_fav.setText(T('fav_add'))
            return
        self.info.setText(f"<b>{it['name']}</b>  [{kind_name(it['kind'])}]<br>" + T('where', w=it['where'] or '-'))
        self.btn_fav.setText(T('fav_del') if it['id'] in self.cfg['favorites'] else T('fav_add'))

    def copy(self, it):
        if not it:
            return
        QtWidgets.QApplication.clipboard().setText(it['name'])
        self.statusBar().showMessage(T('copied', n=it['name']), 8000)

    def toggle_fav(self):
        it = self.current()
        if not it:
            return
        favs = self.cfg['favorites']
        favs.remove(it['id']) if it['id'] in favs else favs.insert(0, it['id'])
        save_config(self.cfg)
        self.refresh_favs()
        self.on_select()

    def refresh_favs(self):
        by_id = {it['id']: it for it in self.items}
        self.favs.show_items([by_id[i] for i in self.cfg['favorites'] if i in by_id])
        self.tabs.setTabText(1, T('tab_favs', n=self.favs.count()))


# ---------------------------------------------------------------- setup.bat 用

def setup():
    """モデルのダウンロード (進捗はコンソールに表示) + スタートメニュー/フォルダにショートカット作成."""
    from huggingface_hub import snapshot_download
    sys.stdout.reconfigure(errors='replace')  # 英語版Windowsのコンソールでも日本語パスで落ちないように
    print('Downloading AI models (about 7GB, first time only)...')
    snapshot_download(IMG_MODEL)
    snapshot_download(TXT_MODEL, allow_patterns=['*.json', 'pytorch_model.bin', 'sentencepiece.bpe.model', 'tokenizer*'],
                      ignore_patterns=['onnx/*'])
    import pythoncom
    from win32com.shell import shell
    from win32com.propsys import propsys, pscon
    pyw = os.path.join(os.path.dirname(sys.executable), 'pythonw.exe')
    for dst in (os.path.join(HERE, '素材さがし.lnk'),
                os.path.expandvars(r'%APPDATA%\Microsoft\Windows\Start Menu\Programs\素材さがし.lnk')):
        link = pythoncom.CoCreateInstance(shell.CLSID_ShellLink, None, pythoncom.CLSCTX_INPROC_SERVER, shell.IID_IShellLink)
        link.SetPath(pyw)
        link.SetArguments(f'"{os.path.abspath(__file__)}"')
        link.SetWorkingDirectory(HERE)
        link.SetIconLocation(os.path.join(HERE, 'icon.ico'), 0)
        link.SetDescription('素材さがし — CLIP STUDIO material search')
        store = link.QueryInterface(propsys.IID_IPropertyStore)  # 実行中のウィンドウをピン留めアイコンにまとめる
        store.SetValue(pscon.PKEY_AppUserModel_ID, propsys.PROPVARIANTType(APP_ID))
        store.Commit()
        link.QueryInterface(pythoncom.IID_IPersistFile).Save(dst, 0)
        print('Shortcut:', dst)


if __name__ == '__main__':
    if '--setup' in sys.argv:
        setup()
        sys.exit(0)
    import ctypes
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
    app = QtWidgets.QApplication(sys.argv)
    app.setWindowIcon(QtGui.QIcon(os.path.join(HERE, 'icon.ico')))
    win = Main()
    win.show()
    sys.exit(app.exec())
