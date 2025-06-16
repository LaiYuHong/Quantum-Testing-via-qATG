import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import SXGate, RZGate, RYGate, RXGate, CXGate, UGate, UnitaryGate
from qiskit.quantum_info import Operator
from qatg import QATG, QATGFault
from qiskit.qasm2 import dumps

# Fault class definitions with original names
class myFault_1(QATGFault):
    def __init__(self):
        super().__init__(SXGate, 0, "gateType: SX, qubits: 0")
    def createOriginalGate(self):
        return SXGate()
    def createFaultyGate(self, _):
        qc = QuantumCircuit(1)
        qc.append(SXGate(), [0])
        qc.append(RZGate(np.pi / 20), [0])
        return UnitaryGate(Operator(qc), label="myFault_1")

class myFault_2(QATGFault):
    def __init__(self, theta):
        super().__init__(RZGate, 0, f"gateType: RZ, qubits: 0, param: {theta}")
        self.theta = theta
    def createOriginalGate(self):
        return RZGate(self.theta)
    def createFaultyGate(self, _):
        qc = QuantumCircuit(1)
        qc.append(RZGate(self.theta), [0])
        qc.append(RYGate(0.1 * self.theta), [0])
        return UnitaryGate(Operator(qc), label="myFault_2")

class myFault_3(QATGFault):
    def __init__(self):
        super().__init__(CXGate, [0, 1], "gateType: CX, qubits: 0-1 with RX faults")
    def createOriginalGate(self):
        return CXGate()
    def createFaultyGate(self, _):
        qc = QuantumCircuit(2)
        qc.append(RXGate(0.1 * np.pi), [0])
        qc.append(CXGate(), [0, 1])
        qc.append(RXGate(-0.1 * np.pi), [0])
        return UnitaryGate(Operator(qc), label="myFault_3")

# General test runner
def run_qatg_test(fault, circuit_size, init_states, filename):
    generator = QATG(
        circuitSize=circuit_size,
        basisSingleQubitGateSet=[UGate],
        circuitInitializedStates={circuit_size: init_states},
        minRequiredStateFidelity=0.1
    )
    configurations = generator.createTestConfiguration([fault])
    print(f"\n=== {fault.description} ===")
    for idx, config in enumerate(configurations):
        print(f"Configuration {idx}:\n{config}\n{config.circuit}")
        try:
            qasm_str = dumps(config.circuit)
            with open(filename, "w") as f:
                f.write(qasm_str)
            print(f"✅ Saved QASM to {filename}")
        except Exception as e:
            print(f"❌ Error exporting QASM: {e}")

# Run tests for all faults
run_qatg_test(myFault_1(), circuit_size=1, init_states=[1, 0], filename="Sx_test.qasm")
run_qatg_test(myFault_2(np.pi / 2), circuit_size=1, init_states=[1, 0], filename="Rz_test.qasm")
run_qatg_test(myFault_3(), circuit_size=2, init_states=[1, 0, 0, 0], filename="Cnot_test.qasm")
