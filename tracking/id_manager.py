from utils.logger import sys_logger

class IDManager:
    """
    Única Fonte da Verdade para geração e atribuição de IDs numéricos no sistema.
    """
    def __init__(self):
        self._next_id = 1

    def get_next_id(self) -> int:
        assigned_id = self._next_id
        self._next_id += 1
        sys_logger.debug(f"[IDManager] Novo ID atribuído: #{assigned_id}")
        return assigned_id

    def register_id(self, existing_id: int):
        """Registra/sincroniza ID existente para prevenir colisões futuras."""
        if existing_id >= self._next_id:
            self._next_id = existing_id + 1

    def reset(self):
        self._next_id = 1
        sys_logger.info("[IDManager] Contador de IDs resetado.")
