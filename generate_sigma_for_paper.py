import subprocess as sp

step_table = {
    (-6,0):27,
    (-5,0):27,
    (-4,0):27,
    (-3,0):26,
    (-2,0):26,
    (-1,0):26,
    (0,0):24,
    (1,0):23,
    (2,0):22,
    (3,0):21,
    
    (-6,2):25,
    (-5,2):24,
    (-4,2):23,
    (-3,2):22,
    (-2,2):21,
    (-1,2):20,
    (0,2):19,
    (1,2):18,
    (2,2):17,
    (3,2):16,
    
    (-6,4):24,
    (-5,4):22,
    (-4,4):20,
    (-3,4):19,
    (-2,4):17,
    (-1,4):16,
    (0,4):16,
    (1,4):14,
    (2,4):13,
    (3,4):12,
}

STEP_TABLE_OFFSET = 2

if __name__ == "__main__":
    for m in range(-6,4):
        for n in (0,2,4):
            cmd = ['python', 'fisherGenerateDataClass_example.py', str(m), str(n), str(10**(-step_table[(m,n)]-STEP_TABLE_OFFSET))]
            print(' '.join(cmd))
            proc = sp.Popen(cmd)
            proc.wait()
            print(f'{m} joined!')

#import report_fisherlens_scattering_results