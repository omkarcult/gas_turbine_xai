import numpy as np
import pandas as pd

def create_simple_turbine_data():
    np.random.seed(42)
    records = []
    
    turbines = ["Turbine-01", "Turbine-02", "Turbine-03"]
    
    for t_name in turbines:
        cycles = 300
        for i in range(cycles):
            # Base physics signals
            temp = 1000 + (i * 0.5) + np.random.normal(0, 5)     # Slowly rises as turbine wears out
            pressure = 15.0 - (i * 0.01) + np.random.normal(0, 0.1) # Slowly drops
            vibration = 1.0 + (i * 0.008) + np.random.normal(0, 0.05) # Slowly increases
            
            # Health Score (100% -> 0%)
            health = max(0, 100 - (i / 3))
            
            # Remaining Useful Life (RUL)
            rul = cycles - i
            
            # Simple Status Label
            if health > 70:
                status = "Normal Operation"
            elif health > 35:
                status = "Minor Degradation"
            else:
                status = "Critical Fault Imminent"
                
            records.append({
                "turbine_id": t_name,
                "cycle": i,
                "temperature": round(temp, 1),
                "pressure": round(pressure, 2),
                "vibration": round(vibration, 2),
                "health_score": round(health, 1),
                "rul_cycles": rul,
                "status": status
            })
            
    df = pd.DataFrame(records)
    df.to_csv("simple_turbine_data.csv", index=False)
    print("✓ Simple dataset generated successfully as 'simple_turbine_data.csv'")

if __name__ == "__main__":
    create_simple_turbine_data()