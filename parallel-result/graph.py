import matplotlib.pyplot as plt
import numpy as np
import glob

# 데이터 수집
threads = []
times = []
matrix_sizes = []
kernel_sizes = []

# 현재 디렉토리의 *threads.log 파일들 읽기
log_files = glob.glob("*threads.log")

for file in log_files:
    with open(file, 'r') as f:
        lines = f.readlines()
        if len(lines) >= 2:
            # 두 번째 줄 파싱: threads,time,matrix_size,kernel_size
            data = lines[1].strip().split(',')
            if len(data) == 4:
                threads.append(int(data[0]))
                times.append(float(data[1]))
                matrix_sizes.append(int(data[2]))
                kernel_sizes.append(int(data[3]))

# 스레드 수로 정렬
sorted_data = sorted(zip(threads, times, matrix_sizes, kernel_sizes))
threads, times, matrix_sizes, kernel_sizes = zip(*sorted_data)

# 스피드업 계산 (1 스레드 기준)
speedup = [times[0] / t for t in times]
efficiency = [s / t for s, t in zip(speedup, threads)]

# 그래프 생성
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# 1. 실행 시간 vs 스레드 수
ax1.plot(threads, times, 'bo-', linewidth=2, markersize=8)
ax1.set_xlabel('Number of Threads')
ax1.set_ylabel('Execution Time (seconds)')
ax1.set_title('Execution Time vs Thread Count')
ax1.grid(True, alpha=0.3)
ax1.set_xscale('log', base=2)

# 2. 스피드업 vs 스레드 수
ax2.plot(threads, speedup, 'ro-', linewidth=2, markersize=8, label='Actual Speedup')
ax2.plot(threads, threads, 'k--', alpha=0.5, label='Ideal Speedup')
ax2.set_xlabel('Number of Threads')
ax2.set_ylabel('Speedup')
ax2.set_title('Speedup vs Thread Count')
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.set_xscale('log', base=2)
ax2.set_yscale('log', base=2)

# 3. 효율성 vs 스레드 수
ax3.plot(threads, efficiency, 'go-', linewidth=2, markersize=8)
ax3.set_xlabel('Number of Threads')
ax3.set_ylabel('Efficiency')
ax3.set_title('Parallel Efficiency vs Thread Count')
ax3.grid(True, alpha=0.3)
ax3.set_xscale('log', base=2)
ax3.set_ylim(0, 1.1)

# 4. 성능 요약 테이블
ax4.axis('tight')
ax4.axis('off')
table_data = []
for i, (t, time, su, eff) in enumerate(zip(threads, times, speedup, efficiency)):
    table_data.append([f'{t}', f'{time:.2f}s', f'{su:.2f}x', f'{eff:.2f}'])

table = ax4.table(cellText=table_data,
                  colLabels=['Threads', 'Time', 'Speedup', 'Efficiency'],
                  cellLoc='center',
                  loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)
ax4.set_title('Performance Summary')

# 전체 제목
matrix_size = int(np.sqrt(matrix_sizes[0]))  # 1410065408 = 37546^2 정도
fig.suptitle(f'Parallel Convolution Performance Analysis\n'
             f'Matrix: {matrix_size}×{matrix_size}, Kernel: 3×3', 
             fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('parallel_performance_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 데이터 출력
print("Performance Results:")
print("Threads | Time (s) | Speedup | Efficiency")
print("-" * 40)
for t, time, su, eff in zip(threads, times, speedup, efficiency):
    print(f"{t:7d} | {time:8.2f} | {su:7.2f} | {eff:10.2f}")