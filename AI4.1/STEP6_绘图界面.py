"""事件切片绘图窗口；使用启动此窗口的Python运行现有STEP6，不训练模型。"""
from datetime import datetime
from pathlib import Path
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import webbrowser

ROOT = Path(__file__).resolve().parent


def build_command(values):
    start, end = int(values['start']), int(values['end'])
    if not 0 <= start <= end:
        raise ValueError('EvtID起止值须满足 0 ≤ 起点 ≤ 终点（包含端点）')
    strict, relaxed = float(values['strict']), float(values['relaxed'])
    if not 0 < relaxed <= strict < 1:
        raise ValueError('阈值须满足 0 < 宽松 ≤ 严格 < 1')
    if values['recovery'] != 'none' and values['model'] != 'last':
        raise ValueError('片段/Hough回收目前只验证过last模型，请选择last')
    command = [sys.executable, '-u', str(ROOT / 'STEP6_事件切片可视化.py'),
               '--event-range', str(start), str(end), '--model', values['model'],
               '--recovery', values['recovery'], '--strict-threshold', str(strict),
               '--relaxed-threshold', str(relaxed)]
    if values['source'].strip():
        command += ['--source', values['source'].strip(), '--split', 'all']
        if values['cell_map'].strip():
            command += ['--cell-map', values['cell_map'].strip()]
        if values['sheet'].strip():
            command += ['--sheet', values['sheet'].strip()]
    else:
        command += ['--split', values['split']]
    if values['track'].strip():
        command += ['--track-id', str(int(values['track']))]
    output = Path(values['output'].strip()).expanduser() if values['output'].strip() else (
        ROOT / 'whole_event_retraining/event_visualizations' / ('interactive_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
    return command + ['--output-dir', str(output.resolve())], output.resolve()


def launch():
    window = tk.Tk()
    window.title('AI4 事件切片可视化')
    window.geometry('1000x720')
    window.minsize(850, 650)
    panel = ttk.Frame(window, padding=14)
    panel.pack(fill='both', expand=True)
    panel.columnconfigure(1, weight=1)
    variables = {name: tk.StringVar(value=value) for name, value in {
        'source': '', 'cell_map': os.environ.get('MACE_CDC_CELL_MAP', 'cdc_cell_map.xlsx'), 'sheet': '',
        'start': '0', 'end': '2', 'split': 'all', 'model': 'last', 'recovery': 'none',
        'strict': '0.70', 'relaxed': '0.575', 'track': '', 'output': ''}.items()}

    def browse(name, directory=False):
        value = filedialog.askdirectory(parent=window) if directory else filedialog.askopenfilename(
            parent=window, filetypes=[('Excel / CSV', '*.xlsx *.csv'), ('所有文件', '*.*')])
        if value:
            variables[name].set(value)

    for row, (label, name) in enumerate([('数据源（留空用原数据集）', 'source'), ('丝几何表', 'cell_map'),
                                        ('Excel工作表（留空取首表）', 'sheet')]):
        ttk.Label(panel, text=label).grid(row=row, column=0, sticky='w', pady=5)
        ttk.Entry(panel, textvariable=variables[name]).grid(row=row, column=1, sticky='ew', padx=8)
        if name != 'sheet':
            buttons = ttk.Frame(panel)
            buttons.grid(row=row, column=2, sticky='w')
            ttk.Button(buttons, text='选文件', command=lambda n=name: browse(n)).pack(side='left')
            if name == 'source':
                ttk.Button(buttons, text='选文件夹', command=lambda: browse('source', True)).pack(side='left', padx=4)

    options = ttk.Frame(panel)
    options.grid(row=3, column=0, columnspan=3, sticky='ew', pady=10)
    for column, (label, name, choices) in enumerate([
        ('起始EvtID', 'start', None), ('结束EvtID（包含）', 'end', None),
        ('原数据划分', 'split', ['all', 'validation', 'train', 'test']),
        ('模型', 'model', ['last', 'best', 'stage1', 'original']),
        ('回收方式', 'recovery', ['none', 'fragments', 'hough'])]):
        ttk.Label(options, text=label).grid(row=0, column=column, sticky='w', padx=5)
        widget = ttk.Combobox(options, textvariable=variables[name], values=choices, state='readonly', width=14) if choices else ttk.Entry(options, textvariable=variables[name], width=16)
        widget.grid(row=1, column=column, padx=5, pady=5)
    for column, (label, name) in enumerate([('严格阈值', 'strict'), ('宽松阈值', 'relaxed'), ('放大TrkID（可空）', 'track')]):
        ttk.Label(options, text=label).grid(row=2, column=column, sticky='w', padx=5)
        ttk.Entry(options, textvariable=variables[name], width=16).grid(row=3, column=column, padx=5, pady=5)
    ttk.Label(panel, text='输出目录（留空自动新建）').grid(row=4, column=0, sticky='w')
    ttk.Entry(panel, textvariable=variables['output']).grid(row=4, column=1, sticky='ew', padx=8)
    ttk.Button(panel, text='选文件夹', command=lambda: browse('output', True)).grid(row=4, column=2, sticky='w')
    ttk.Label(panel, text='外部数据：读取文件夹直接包含的CSV/XLSX，按EvtID整事件构图；固定筛选Good=True / Axial / ExtendedDecayWithSpin。\n'
              '外部数据不沿用旧训练/验证划分；d直接使用表中数值，不重复加误差。模型last=62轮，best=34轮。',
              wraplength=940).grid(row=5, column=0, columnspan=3, sticky='w', pady=10)
    status = tk.StringVar(value=f'当前Python：{sys.executable}')
    ttk.Label(panel, textvariable=status, wraplength=940).grid(row=6, column=0, columnspan=3, sticky='w')
    actions = ttk.Frame(panel)
    actions.grid(row=7, column=0, columnspan=3, sticky='w', pady=8)
    log = scrolledtext.ScrolledText(panel, height=15, wrap='word', state='disabled')
    log.grid(row=8, column=0, columnspan=3, sticky='nsew')
    panel.rowconfigure(8, weight=1)
    messages = queue.Queue()
    process = None
    last_output = None

    def append(text):
        log.configure(state='normal')
        log.insert('end', text)
        log.see('end')
        log.configure(state='disabled')

    def start():
        nonlocal process, last_output
        try:
            command, output = build_command({key: var.get() for key, var in variables.items()})
            process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding='utf-8', errors='replace',
                                       env={**os.environ, 'PYTHONIOENCODING': 'utf-8'},
                                       creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        except (ValueError, OSError) as exc:
            messagebox.showerror('无法开始', str(exc), parent=window)
            return
        last_output = output
        start_button.configure(state='disabled')
        stop_button.configure(state='normal')
        open_button.configure(state='disabled')
        status.set('正在读取与绘图；大型Excel扫描可能需要几分钟。')
        append('\n' + subprocess.list2cmdline(command) + '\n')

        def collect(child):
            try:
                for line in child.stdout:
                    messages.put(('log', line))
            finally:
                child.stdout.close()
                messages.put(('done', child.wait()))
        threading.Thread(target=collect, args=(process,), daemon=True).start()

    def stop():
        if process is not None and process.poll() is None:
            process.terminate()
            status.set('正在停止；已保存的图会保留。')

    def poll():
        nonlocal process
        try:
            while True:
                kind, value = messages.get_nowait()
                if kind == 'log':
                    append(value)
                else:
                    process = None
                    start_button.configure(state='normal')
                    stop_button.configure(state='disabled')
                    ready = value == 0 and last_output is not None and (last_output / 'index.html').exists()
                    open_button.configure(state='normal' if ready else 'disabled')
                    status.set(f'完成：{last_output}' if ready else '未完成，请查看下方错误信息。若缺少numpy等包，请用pytorch环境启动此窗口。')
        except queue.Empty:
            pass
        window.after(150, poll)

    def close():
        if process is not None and process.poll() is None:
            messagebox.showinfo('绘图进行中', '请先点击停止，等任务结束后再关闭窗口。', parent=window)
        else:
            window.destroy()

    start_button = ttk.Button(actions, text='开始绘图', command=start)
    start_button.pack(side='left')
    stop_button = ttk.Button(actions, text='停止', command=stop, state='disabled')
    stop_button.pack(side='left', padx=8)
    open_button = ttk.Button(actions, text='打开结果', state='disabled',
                             command=lambda: webbrowser.open((last_output / 'index.html').as_uri()))
    open_button.pack(side='left')
    window.protocol('WM_DELETE_WINDOW', close)
    poll()
    window.mainloop()


if __name__ == '__main__':
    launch()
