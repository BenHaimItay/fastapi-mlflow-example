from app.inference_service import InferenceService

server = InferenceService()
app = server.app


def main() -> None:
    server.listen()


if __name__ == "__main__":
    main()
