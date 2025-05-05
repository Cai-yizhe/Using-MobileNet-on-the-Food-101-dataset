import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import ImageTk
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
import sys
import datetime
import subprocess


# 尝试设置中文字体
def setup_chinese_fonts():
    """设置支持中文的字体"""
    try:
        # 检查系统中可用的字体
        font_list = subprocess.check_output(['fc-list', ':lang=zh', 'file']).decode('utf-8').split('\n')
        if font_list:
            print(f"发现的中文字体: {font_list[:3]}")

            # 常见的中文字体名称
            chinese_fonts = [
                'Noto Sans CJK SC', 'Noto Sans CJK TC', 'WenQuanYi Zen Hei',
                'WenQuanYi Micro Hei', 'SimSun', 'SimHei', 'Microsoft YaHei',
                'PingFang SC', 'STHeiti', 'Source Han Sans CN', 'Source Han Serif CN',
                'Droid Sans Fallback'
            ]

            # 尝试设置字体
            for font in chinese_fonts:
                for system_font in font_list:
                    if font.lower() in system_font.lower():
                        print(f"使用中文字体: {font}")
                        matplotlib.rcParams['font.sans-serif'] = [font, 'DejaVu Sans', 'sans-serif']
                        matplotlib.rcParams['axes.unicode_minus'] = False
                        return font

        # 如果没有找到指定中文字体，使用系统默认
        print("未找到指定的中文字体，使用系统默认字体")
        return None
    except Exception as e:
        print(f"设置中文字体时出错: {e}")
        return None


# 设置中文字体
chinese_font = setup_chinese_fonts()

# 设置matplotlib
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 加载训练好的模型
print("正在加载模型...")
model_path = 'food101_mobilenet_final.h5'
try:
    model = load_model(model_path)
    print("模型加载成功")
except Exception as e:
    print(f"加载模型出错: {str(e)}")
    sys.exit(1)

# 加载类别信息
meta_dir = 'archive/meta/meta'
try:
    with open(os.path.join(meta_dir, 'classes.txt'), 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    print(f"已加载{len(classes)}个食品类别")
except Exception as e:
    print(f"加载类别信息出错: {str(e)}")
    sys.exit(1)


class FoodClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("食品分类测试工具")
        self.root.geometry("900x700")
        self.root.configure(bg='white')

        # 检测系统字体
        self.detect_fonts()

        # 历史记录
        self.history = []

        self.setup_ui()

    def detect_fonts(self):
        """检测系统字体"""
        # 尝试不同的中文字体名称
        self.font_family = "TkDefaultFont"
        self.font_size = 12

        # 尝试常见中文字体
        chinese_fonts = [
            'Noto Sans CJK SC', 'Noto Sans CJK TC', 'WenQuanYi Zen Hei',
            'WenQuanYi Micro Hei', 'SimSun', 'SimHei', 'Microsoft YaHei',
            'PingFang SC', 'STHeiti', 'Source Han Sans CN', 'Source Han Serif CN',
            'Droid Sans Fallback'
        ]

        for font in chinese_fonts:
            try:
                # 尝试创建一个带字体的标签
                test_label = tk.Label(self.root, text="测试", font=(font, self.font_size))
                test_label.destroy()
                self.font_family = font
                print(f"使用中文字体: {font}")
                break
            except:
                continue

        print(f"GUI使用字体: {self.font_family}")

    def setup_ui(self):
        # 主标题
        title_label = tk.Label(self.root, text="食品图像分类测试工具",
                               font=(self.font_family, 20, "bold"), bg='white')
        title_label.pack(pady=10)

        # 创建顶部框架用于显示图像和结果
        top_frame = tk.Frame(self.root, bg='white')
        top_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # 左侧显示图像
        self.image_frame = tk.LabelFrame(top_frame, text="输入图像", bg='white', font=(self.font_family, 12))
        self.image_frame.pack(side=tk.LEFT, fill='both', expand=True, padx=5, pady=5)

        self.image_label = tk.Label(self.image_frame, bg='white')
        self.image_label.pack(padx=10, pady=10, fill='both', expand=True)

        # 右侧显示预测结果
        result_frame = tk.LabelFrame(top_frame, text="预测结果", bg='white', font=(self.font_family, 12))
        result_frame.pack(side=tk.RIGHT, fill='both', expand=True, padx=5, pady=5)

        # 图表
        self.figure = plt.Figure(figsize=(5, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.chart_canvas = FigureCanvasTkAgg(self.figure, result_frame)
        self.chart_canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)

        # 结果标签
        self.result_label = tk.Label(self.root, text="请选择一张图像开始测试",
                                     font=(self.font_family, 14), bg='white')
        self.result_label.pack(pady=5)

        # 状态标签
        self.status_label = tk.Label(self.root, text="就绪",
                                     font=(self.font_family, 10), bg='white', fg='green')
        self.status_label.pack(pady=5)

        # 按钮框架
        button_frame = tk.Frame(self.root, bg='white')
        button_frame.pack(pady=10)

        # 选择图像按钮
        self.select_button = tk.Button(button_frame, text="选择图像",
                                       command=self.load_and_predict,
                                       font=(self.font_family, 12), bg='#4CAF50', fg='white', padx=10)
        self.select_button.pack(side=tk.LEFT, padx=5)

        # 清除按钮
        self.clear_button = tk.Button(button_frame, text="清除显示",
                                      command=self.clear_display,
                                      font=(self.font_family, 12), bg='#2196F3', fg='white', padx=10)
        self.clear_button.pack(side=tk.LEFT, padx=5)

        # 保存结果按钮
        self.save_button = tk.Button(button_frame, text="保存结果",
                                     command=self.save_results,
                                     font=(self.font_family, 12), bg='#FF9800', fg='white', padx=10)
        self.save_button.pack(side=tk.LEFT, padx=5)
        self.save_button.config(state=tk.DISABLED)  # 初始禁用

        # 退出按钮
        self.exit_button = tk.Button(button_frame, text="退出",
                                     command=self.root.destroy,
                                     font=(self.font_family, 12), bg='#F44336', fg='white', padx=10)
        self.exit_button.pack(side=tk.LEFT, padx=5)

        # 历史记录框架
        history_frame = tk.LabelFrame(self.root, text="测试历史", bg='white', font=(self.font_family, 12))
        history_frame.pack(fill='both', expand=False, padx=10, pady=10)

        # 历史记录列表
        self.history_listbox = tk.Listbox(history_frame, height=4, width=80, font=(self.font_family, 10))
        self.history_listbox.pack(side=tk.LEFT, fill='both', expand=True, padx=5, pady=5)

        # 滚动条
        scrollbar = tk.Scrollbar(history_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 连接列表和滚动条
        self.history_listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.history_listbox.yview)

        # 版本信息
        version_label = tk.Label(self.root, text=f"TensorFlow: {tf.__version__}, Python: {sys.version.split()[0]}",
                                 font=(self.font_family, 8), bg='white', fg='gray')
        version_label.pack(side=tk.BOTTOM, pady=5)

    def load_and_predict(self):
        """加载图像并进行预测"""
        self.status_label.config(text="正在选择图像...", fg='blue')
        self.root.update()

        # 打开文件选择对话框
        file_path = filedialog.askopenfilename(
            title="选择食品图像",
            filetypes=[("图像文件", "*.jpg *.jpeg *.png *.bmp")]
        )

        if not file_path:
            self.status_label.config(text="已取消选择", fg='orange')
            return

        try:
            self.status_label.config(text=f"正在处理: {os.path.basename(file_path)}", fg='blue')
            self.root.update()

            # 加载并显示图像
            img = Image.open(file_path).convert('RGB')
            img_display = img.copy()
            img_display.thumbnail((300, 300))
            tk_img = ImageTk.PhotoImage(img_display)
            self.image_label.config(image=tk_img)
            self.image_label.image = tk_img

            # 预处理图像
            img_resized = img.resize((224, 224))
            img_array = np.array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # 预测
            self.status_label.config(text="正在预测...", fg='blue')
            self.root.update()
            predictions = model.predict(img_array, verbose=0)

            # 获取前5个预测结果
            top_indices = np.argsort(predictions[0])[::-1][:5]
            top_classes = [classes[i] for i in top_indices]
            top_probs = [predictions[0][i] for i in top_indices]

            # 显示结果标签
            result_text = f"预测结果: {top_classes[0]} (概率: {top_probs[0] * 100:.2f}%)"
            self.result_label.config(text=result_text)

            # 绘制结果图表
            self.ax.clear()
            colors = ['#4CAF50', '#8BC34A', '#CDDC39', '#FFEB3B', '#FFC107']
            y_pos = np.arange(len(top_classes))
            bars = self.ax.barh(y_pos, top_probs, color=colors)
            self.ax.set_yticks(y_pos)
            self.ax.set_yticklabels(top_classes)
            self.ax.set_xlim([0, 1])
            self.ax.set_xlabel('概率')
            self.ax.set_title('预测结果')

            # 添加概率值文本
            for i, bar in enumerate(bars):
                self.ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                             f'{top_probs[i] * 100:.1f}%', va='center')

            self.figure.tight_layout()
            self.chart_canvas.draw()

            # 更新状态和启用保存按钮
            self.status_label.config(text="预测完成", fg='green')
            self.save_button.config(state=tk.NORMAL)

            # 添加到历史记录
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            history_entry = f"{timestamp} - {os.path.basename(file_path)} → {top_classes[0]} ({top_probs[0] * 100:.2f}%)"
            self.history.append({
                'timestamp': timestamp,
                'image_path': file_path,
                'prediction': top_classes[0],
                'probability': top_probs[0],
                'all_predictions': list(zip(top_classes, top_probs))
            })
            self.history_listbox.insert(0, history_entry)

        except Exception as e:
            self.status_label.config(text=f"错误: {str(e)}", fg='red')
            messagebox.showerror("错误", f"发生错误: {str(e)}")
            print(f"错误详情: {str(e)}")

    def clear_display(self):
        """清除显示的图像和预测结果"""
        self.image_label.config(image='')
        self.ax.clear()
        self.chart_canvas.draw()
        self.result_label.config(text="请选择一张图像开始测试")
        self.status_label.config(text="就绪", fg='green')
        self.save_button.config(state=tk.DISABLED)

    def save_results(self):
        """保存预测结果"""
        if not self.history:
            messagebox.showinfo("提示", "没有结果可保存")
            return

        # 创建保存对话框
        file_path = filedialog.asksaveasfilename(
            title="保存结果",
            defaultextension=".png",
            filetypes=[("PNG图像", "*.png"), ("所有文件", "*.*")]
        )

        if not file_path:
            return

        try:
            # 保存当前图表
            self.figure.savefig(file_path, dpi=100)
            self.status_label.config(text=f"结果已保存到 {os.path.basename(file_path)}", fg='green')
        except Exception as e:
            messagebox.showerror("错误", f"保存结果失败: {str(e)}")


# 创建备用命令行版本的函数(当GUI无法正常工作时使用)
def run_command_line_version():
    print("\n=== 食品分类测试工具(命令行版) ===")
    print("图形界面初始化失败，转为使用命令行版本")

    def predict_single_image(image_path):
        """预测单张图像"""
        # 加载图像
        img = Image.open(image_path).convert('RGB')

        # 调整大小
        img_resized = img.resize((224, 224))

        # 转换为数组并归一化
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # 模型预测
        predictions = model.predict(img_array, verbose=0)

        # 获取预测类别和概率
        top_indices = np.argsort(predictions[0])[::-1][:5]
        top_classes = [classes[i] for i in top_indices]
        top_probs = [predictions[0][i] for i in top_indices]

        # 显示原始图像和预测结果
        plt.figure(figsize=(12, 6))

        # 显示图像
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title('输入图像')
        plt.axis('off')

        # 显示预测结果
        plt.subplot(1, 2, 2)
        colors = ['#4CAF50', '#8BC34A', '#CDDC39', '#FFEB3B', '#FFC107']
        bars = plt.barh(np.arange(len(top_classes)), top_probs, color=colors)
        plt.yticks(np.arange(len(top_classes)), top_classes)
        plt.xlabel('概率')
        plt.title('预测结果')
        plt.xlim([0, 1])

        # 添加概率值文本
        for i, bar in enumerate(bars):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                     f'{top_probs[i] * 100:.1f}%', va='center')

        plt.tight_layout()

        # 生成唯一文件名
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = f'prediction_result_{timestamp}.png'
        plt.savefig(result_file, dpi=100)
        print(f"\n结果已保存为图像: {result_file}")

        # 显示图像窗口
        plt.show()

        # 打印结果
        print("\n预测结果:")
        for i, (cls, prob) in enumerate(zip(top_classes, top_probs)):
            print(f"{i + 1}. {cls}: {prob:.4f} ({prob * 100:.2f}%)")

        return top_classes[0], top_probs[0]

    # 循环测试
    while True:
        print("\n" + "-" * 50)
        image_path = input("\n请输入图像路径(输入'q'或'退出'结束): ")

        if image_path.lower() in ['q', 'quit', '退出', 'exit']:
            print("\n测试结束，谢谢使用!")
            break

        if not os.path.exists(image_path):
            print(f"错误: 文件 '{image_path}' 不存在")
            continue

        try:
            print(f"正在处理图像: {os.path.basename(image_path)}...")
            top_class, top_prob = predict_single_image(image_path)
            print(f"\n最可能的类别是: {top_class}")
            print(f"概率: {top_prob * 100:.2f}%")
        except Exception as e:
            print(f"预测过程中出错: {str(e)}")


# 运行应用
if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = FoodClassifierApp(root)
        root.mainloop()
    except Exception as e:
        print(f"启动GUI失败: {str(e)}")
        print("切换到命令行模式...")
        run_command_line_version()